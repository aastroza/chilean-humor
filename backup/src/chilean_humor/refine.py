import instructor
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

class SequentialOutcome(Enum):
    CONTINUATION = "continuation"
    NOT_CONTINUATION = "not_continuation"

class SequentialAnalysis(BaseModel):
    reasoning: str = Field(
        description="Reasoning behind the analysis.", exclude=True
    )
    outcome: SequentialOutcome

SYSTEM_PROMPT = """You are given two sequential texts from the transcriptions of comedians' routines. Your task is to analyze these texts and determine if the second text is a continuation of the first one or if it introduces a different theme or topic. Pay attention to context, content, and the logical flow of ideas.
In your analysis, consider the following:

* Summarize the main theme or topic covered in each text excerpt.
* Identify any common subjects, jokes, or recurring elements that may indicate a continuation of the same theme/topic.
* Note any significant shifts in tone, style, or subject matter that suggest a different theme/topic in the second text.
* Based on your observations, provide your assessment of whether the second text is a continuation of the first or covers a different theme/topic.
"""

def detect_continuity(text1: str, text2: str) -> SequentialAnalysis:
    client = instructor.from_openai(OpenAI())

    try:
        analysis = client.chat.completions.create(
            model="gpt-4o",
            response_model=SequentialAnalysis,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"first text: {text1}"},
                {"role": "user", "content": f"second text: {text2}"},
            ],
        )
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing continuity: {e}")       
        return SequentialAnalysis(outcome=SequentialOutcome.DIFERRENT_TOPIC)