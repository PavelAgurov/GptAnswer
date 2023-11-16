"""
    Step back strategy
    https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb
"""
# pylint: disable=C0301,C0103,C0304,C0303,W0611,W0511,R0913,R0402,W1203

import logging
import traceback
from dataclasses import dataclass

from langchain.callbacks import get_openai_callback
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

logger : logging.Logger = logging.getLogger()

step_back_prompt_template = """\
You are an expert at world knowledge. 
Your task is to step back and paraphrase a question to a more generic 
step-back question, which is easier to answer.

<question>
{question}
</question>
"""

@dataclass
class StepbackResult():
    """Result of step back"""
    answer      : str
    tokens_used : int
    error       : bool

class StepbackChain():
    """Step back chain"""
    
    def __init__(self, llm):
        step_back_prompt = PromptTemplate(template= step_back_prompt_template, input_variables=["question"])
        self.step_back_chain = LLMChain(llm= llm, prompt= step_back_prompt)

    def run(self, question : str) -> StepbackResult:
        """Run step back"""
        tokens_used = 0

        logger.info(f"Run step back question: [{question}]")
        try:
            with get_openai_callback() as cb:
                answer_result = self.step_back_chain.run(question = question)
            tokens_used += cb.total_tokens
            logger.debug(answer_result)
            return StepbackResult(answer_result, tokens_used, False)
        except Exception as error: # pylint: disable=W0718
            logger.exception(error)
            logger.error(traceback.format_exc())
            return StepbackResult("", tokens_used, True)
