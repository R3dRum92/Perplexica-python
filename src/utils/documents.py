import requests
from bs4 import BeautifulSoup
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.logger import logger
from typing import List
from io import BytesIO


def get_documents_from_links(links: List[str]):
    splitter = RecursiveCharacterTextSplitter()

    docs = []

    for link in links:
        if not link.startswith(("http://", "https://")):
            link = f"https://{link}"

        try:
            response = requests.get(link)
            response.raise_for_status()

            if response.headers["Content-Type"] == "application/pdf":
                with pdfplumber.open(BytesIO(response.content)) as pdf:
                    pdf_text = ""
                    for page in pdf.pages:
                        pdf_text += page.extract_text()

                    parsed_text = " ".join(pdf_text.split())
                    splitted_text = splitter.split_text(parsed_text)
                    title = "PDF Document"

                    link_docs = [
                        Document(
                            page_content=text, metadata={"title": title, "url": link}
                        )
                        for text in splitted_text
                    ]
                    docs.extend(link_docs)

            else:
                soup = BeautifulSoup(response.text, "html.parser")
                parsed_text = " ".join(
                    [element.get_text() for element in soup.find_all("p")]
                )

                parsed_text = (
                    parsed_text.replace("\r\n", " ")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .strip()
                )

                splitted_text = splitter.split_text(parsed_text)

                title_tag = soup.find("title")
                title = title_tag.text if title_tag else link

                link_docs = [
                    Document(page_content=text, metadata={"title": title, "url": link})
                    for text in splitted_text
                ]
                docs.extend(link_docs)

        except Exception as e:
            logger.error(f"Error at generating documents from link {link}: {str(e)}")
            docs.append(
                Document(
                    page_content=f"Failed to retrieve content from the link: {str(e)}",
                    metadata={"title": "Failed to retrieve content", "url": link},
                )
            )

    return docs


async def main():
    links = [
        "http://localhost:7000/Fourier_Analysis-1.pdf",
        "http://localhost:7000/Manchester_Encoding.pdf",
    ]
    docs = get_documents_from_links(links)

    for doc in docs:
        print(f"Title: {doc.metadata['title']}")
        print(f"URL: {doc.metadata['url']}")
        print(
            f"Content: {doc.page_content}..."
        )  # Show first 100 characters of the content


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
