from langchain_core.output_parsers.base import BaseOutputParser
import re
from typing import List, Optional
from pydantic import Field


class LineListOutputParser(BaseOutputParser[List[str]]):
    key: str = Field(default="questions")

    @classmethod
    def lc_name(cls) -> str:
        return "LineListOutputParser"

    @property
    def lc_namespace(self) -> List[str]:
        return ["langchain", "output_parsers", "line_list_output_parser"]

    def parse(self, text: str) -> List[str]:
        regex = r"^(\s*(-|\*|\d+\.\s|\d+\)\s|\u2022)\s*)+"

        start_tag = f"<{self.key}>"
        end_tag = f"</{self.key}>"

        start_index = text.find(start_tag)
        end_index = text.find(end_tag)

        if start_index == -1 and end_index == -1:
            return []

        if start_index != -1:
            content_start = start_index + len(start_tag)
        else:
            content_start = 0

        if end_index != -1:
            content_end = end_index
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()

        lines = content.split("\n")

        parsed_lines = [
            re.sub(regex, "", line).strip() for line in lines if line.strip()
        ]

        return parsed_lines

    async def aparse(self, text: str) -> List[str]:
        return self.parse(text)

    def get_format_instructions(self):
        return (
            "Provide a list of items enclosed within "
            f"<{self.key}> and </{self.key}> tags, each separated by a newline. "
            "Each item should start with a list marker like '-', '*', '1.', or '•'."
        )
