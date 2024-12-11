from langchain_core.output_parsers.base import BaseOutputParser
import re
from typing import Optional, List
from pydantic import Field


class LineOutputParser(BaseOutputParser):
    key: str = Field(default="questions")

    @classmethod
    def lc_name():
        return "LineOutputParser"

    @property
    def lc_namespace(self) -> List[str]:
        return ["langchain", "output_parsers", "line_output_parser"]

    async def parse(self, text: str) -> str:
        regex = r"^\s*(-|\*|\d+\.\s|\d+\)\s|\u2022)\s*"

        start_key_index = text.find(f"<{self.key}>")
        end_key_index = text.find(f"</{self.key}>")

        if start_key_index == -1 or end_key_index == -1:
            return ""

        questions_start_index = (
            start_key_index + len(f"<{self.key}>") if start_key_index != -1 else 0
        )
        questions_end_index = end_key_index if end_key_index != -1 else len(text)

        line = text[questions_start_index:questions_end_index].strip()

        line = re.sub(regex, "", line)

        return line

    def get_format_instructions(self):
        pass
