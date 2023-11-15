"""
    Backend
"""

# pylint: disable=C0301,C0103,C0303,C0411,C0304

import logging
from dataclasses import dataclass

logger : logging.Logger = logging.getLogger()

@dataclass
class Answer:
    """Class for LLM answer"""
    answer      : str
    tokens_used : int

class Backend():
    """Backend class"""
    

    def get_answer(self, question : str) -> Answer:
        """Generate answer"""
        return Answer(f'!!{question}!!', 100)