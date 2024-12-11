import os
import json
import asyncio
import pathlib
import shutil
import datetime
from typing import List, Any, Dict
from utils.logger import logger
from utils.documents import get_documents_from_links
from utils.compute_similarity import compute_similarity
from utils.format_history import format_chat_history_as_string
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from lib.output_parsers.list_line_output_parser import LineListOutputParser
from lib.output_parsers.line_output_parser import LineOutputParser
from langchain_core.documents import Document
from lib.searxng import search_searxng, SearxngSearchOptions
from langchain_core.runnables.schema import StreamEvent
from dataclasses import dataclass, field
import eventlet


class EventEmitter:
    def __init__(self):
        self.events = {}

    def on(self, event_name: str, callback):
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(callback)

    def emit(self, event_name: str, data: Any):
        if event_name in self.events:
            for callback in self.events[event_name]:
                callback(data)


@dataclass
class Config:
    search_web: bool = True
    rerank: bool = True
    summarizer: bool = True
    rerank_threshold: float = 0.3
    query_generator_prompt: str = ""
    response_prompt: str = ""
    active_engines: List[str] = field(default_factory=list)


class MetaSearchAgent:
    def __init__(self, config: Config):
        self.config = config
        self.str_parser = StrOutputParser()

    async def create_search_retriever_chain(self, llm: BaseChatModel):
        llm.temperature = 0
        runnable_sequence = (
            PromptTemplate.from_template(self.config.query_generator_prompt)
            | llm
            | self.str_parser
            | RunnableLambda(self.parse_links)
        )

        return runnable_sequence

    async def parse_links(self, input_text: str, llm: BaseChatModel) -> Dict[str, Any]:
        links_parser = LineListOutputParser(key="Links")
        question_parser = LineOutputParser(key="question")

        links = await links_parser.parse(input_text)
        question = (
            await question_parser.parse(input_text)
            if self.config.summarizer
            else input_text
        )

        if question == "not_needed":
            return {"query": "", "docs": []}

        if links:
            question = question if question else "summarize"
            docs = []

            link_docs = get_documents_from_links({"links": links})
            doc_groups = []

            for doc in link_docs:
                existing_doc = (
                    next(
                        d
                        for d in doc_groups
                        if d.metadata["url"] == doc.metadata["url"]
                        and d.metadata.get("totalDocs", 0) < 10
                    ),
                    None,
                )
                if not existing_doc:
                    new_doc = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "totalDocs": 1},
                    )
                    doc_groups.append(new_doc)
                else:
                    existing_doc.page_content += f"\n\n{doc.page_content}"
                    existing_doc.metadata["totalDocs"] += 1

            summarized_docs = await self.summarize_documents(doc_groups, question, llm)
            docs.extend(summarized_docs)
            return {"query": question, "docs": docs}
        else:
            res = await search_searxng(
                question,
                SearxngSearchOptions(language="en", engines=self.config.active_engines),
            )

            documents = [
                Document(
                    page_content=result["content"],
                    metadata={
                        "title": result["title"],
                        "url": result["url"],
                        **(
                            {"img_src": result["img_src"]}
                            if "img_src" in result
                            else {}
                        ),
                    },
                )
                for result in res.get("results", [])
            ]

            return {"query": question, "docs": documents}

    async def summarize_documents(
        self, doc_groups: List[Document], question: str, llm: BaseChatModel
    ) -> List[Document]:
        async def summarize_doc(doc: Document) -> Document:
            prompt = f"""
            You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
            text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
            If the query is "summarize", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.

            - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
            - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
            - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.

            The text will be shared inside the `text` XML tag, and the query inside the `query` XML tag.

            <example>
            1. `<text>
            Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers. 
            It was first released in 2013 and is developed by Docker, Inc. Docker is designed to make it easier to create, deploy, and run applications 
            by using containers.
            </text>

            <query>
            What is Docker and how does it work?
            </query>

            Response:
            Docker is a revolutionary platform-as-a-service product developed by Docker, Inc., that uses container technology to make application 
            deployment more efficient. It allows developers to package their software with all necessary dependencies, making it easier to run in 
            any environment. Released in 2013, Docker has transformed the way applications are built, deployed, and managed.
            `
            2. `<text>
            The theory of relativity, or simply relativity, encompasses two interrelated theories of Albert Einstein: special relativity and general
            relativity. However, the word "relativity" is sometimes used in reference to Galilean invariance. The term "theory of relativity" was based
            on the expression "relative theory" used by Max Planck in 1906. The theory of relativity usually encompasses two interrelated theories by
            Albert Einstein: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity.
            General relativity explains the law of gravitation and its relation to other forces of nature. It applies to the cosmological and astrophysical
            realm, including astronomy.
            </text>

            <query>
            summarize
            </query>

            Response:
            The theory of relativity, developed by Albert Einstein, encompasses two main theories: special relativity and general relativity. Special
            relativity applies to all physical phenomena in the absence of gravity, while general relativity explains the law of gravitation and its
            relation to other forces of nature. The theory of relativity is based on the concept of "relative theory," as introduced by Max Planck in
            1906. It is a fundamental theory in physics that has revolutionized our understanding of the universe.
            `
            </example>

            Everything below is the actual data you will be working with. Good luck!

            <query>
            {question}
            </query>

            <text>
            {doc.page_content}
            </text>

            Make sure to answer the query in the summary.
            """

            res = await llm.invoke(prompt)

            summarized_doc = Document(
                page_content=res.content,
                metadata={
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                },
            )
            return summarized_doc

        tasks = [summarize_doc(doc) for doc in doc_groups]

        summarized_docs = await asyncio.gather(*tasks)

        return summarized_docs

    async def create_answering_chain(
        self,
        llm: BaseChatModel,
        file_ids: List[str],
        embeddings: Embeddings,
        optimization_mode: str,
    ):
        async def retrieve_context(input_data: Dict[str, Any]) -> List[Document]:
            processed_history = format_chat_history_as_string(
                input_data["chat_history"]
            )
            query = input_data["query"]
            docs = None

            if self.config.search_web:
                search_retriever_chain = await self.create_search_retriever_chain(llm)
                search_result = await search_retriever_chain.invoke(
                    {"chat_history": processed_history, "query": query}
                )
                query = search_result["query"]
                docs = search_result["docs"]

            sorted_docs = await self.rerank_docs(
                query, docs or [], file_ids, embeddings, optimization_mode
            )

            return sorted_docs

        runnable_map = RunnableMap(
            {
                "query": lambda x: x["query"],
                "chat_history": lambda x: x["chat_history"],
                "date": lambda x: datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "context": RunnableLambda(retrieve_context).pipe(self.process_docs),
            }
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.response_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{query}"),
            ]
        )

        runnable_sequence = runnable_map | prompt | llm | self.str_parser

        return runnable_sequence

    async def rerank_docs(
        self,
        query: str,
        docs: List[Document],
        file_ids: List[str],
        embeddings: Embeddings,
        optimization_mode: str,
    ) -> List[Document]:
        if not docs and not file_ids:
            return []

        files_data = []
        for file_id in file_ids:
            file_path = pathlib.Path(os.path.join(os.getcwd(), "uploads", file_id))
            content_path = file_path.with_suffix(".json")
            embeddings_path = file_path.with_suffix(".embeddings.json")

            with open(content_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            with open(embeddings_path, "r", encoding="utf-8") as f:
                embeddings_data = json.load(f)

            file_similarity_search_object = [
                {
                    "fileName": content["title"],
                    "content": c,
                    "embeddings": embeddings_data["embeddings"][i],
                }
                for i, c in enumerate(content["contents"])
            ]
            files_data.extend(file_similarity_search_object)

        if query.lower() == "summarize":
            return docs[:15]

        docs_with_content = [doc for doc in docs if doc.page_content]

        if optimization_mode == "speed" or not self.config.rerank:
            if files_data:
                query_embedding = await embeddings.embed_query(query)
                file_docs = [
                    Document(
                        page_content=file_data["content"],
                        metadata={"title": file_data["fileName"], "url": "File"},
                    )
                    for file_data in files_data
                ]

                similarity = [
                    {
                        "index": i,
                        "similarity": compute_similarity(
                            query_embedding, file_data["embeddings"]
                        ),
                    }
                    for i, file_data in enumerate(files_data)
                ]

                sorted_docs = sorted(
                    filter(
                        lambda sim: sim["similarity"]
                        > (self.config.rerank_threshold or 0.3),
                        similarity,
                    ),
                    key=lambda x: x["similarity"],
                    reverse=True,
                )[:15]

                sorted_docs = [file_docs[sim["index"]] for sim in sorted_docs]
                sorted_docs = sorted_docs[:8] if docs_with_content else sorted_docs
                return sorted_docs + docs_with_content[: 15 - len(sorted_docs)]
            else:
                return docs_with_content[:15]
        elif optimization_mode == "balanced":
            doc_embeddings = await embeddings.embed_documents(
                [doc.page_content for doc in docs_with_content]
            )
            query_embedding = await embeddings.embed_query(query)

            all_docs = docs_with_content.copy()
            all_docs += [
                Document(
                    page_content=file_data["content"],
                    metadata={"title": file_data["fileName"], "url": "File"},
                )
                for file_data in files_data
            ]

            all_embeddings = doc_embeddings.copy()
            all_embeddings += [file_data["embeddings"] for file_data in files_data]

            similarity = [
                {"index": i, "similarity": compute_similarity(query_embedding, emb)}
                for i, emb in enumerate(all_embeddings)
            ]

            sorted_docs = sorted(
                filter(
                    lambda sim: sim["similarity"]
                    > (self.config.rerank_threshold or 0.3),
                    similarity,
                ),
                key=lambda x: x["similarity"],
                reverse=True,
            )[:15]

            return [all_docs[sim["index"]] for sim in sorted_docs]
        else:
            return docs_with_content[:15]

    def process_docs(self, docs: List[Document]) -> str:
        return "\n".join([f"{i + 1}. {doc.page_content}" for i, doc in enumerate(docs)])

    async def search_and_answer(
        self,
        message: str,
        history: List[BaseMessage],
        llm: BaseChatModel,
        embeddings: Embeddings,
        optimization_mode: str,
        file_ids: List[str],
    ) -> EventEmitter:
        emitter = EventEmitter()

        answering_chain = await self.create_answering_chain(
            llm, file_ids, embeddings, optimization_mode
        )

        stream = answering_chain.stream_events(
            {"chat_history": history, "query": message}, version="v1"
        )

        await self.handle_stream(stream, emitter)
        return emitter

    async def handle_stream(self, stream, emitter):
        async for event in stream:
            if (
                event["event"] == "on_chain_end"
                and event["name"] == "FinalSourceRetriever"
            ):
                emitter.emit(
                    "data",
                    json.dumps({"type": "sources", "data": event["data"]["output"]}),
                )
            if (
                event["event"] == "on_chain_stream"
                and event["name"] == "FinalResponseGenerator"
            ):
                emitter.emit(
                    "data",
                    json.dumps({"type": "response", "data": event["data"]["chunk"]}),
                )
            if (
                event["event"] == "on_chain_end"
                and event["name"] == "FinalResponseGenerator"
            ):
                emitter.emit("end")
