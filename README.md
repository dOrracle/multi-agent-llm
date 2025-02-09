# LLM based Multi-Agent methods

Welcome to the LLM based Multi-Agent repository! This repository provides a lean implementation of cutting-edge techniques and methods for leveraging multi-agent architectures with Large Language Models (LLMs) for various tasks. This includes methods developed by Agnostiq Inc. as well as other state-of-the-art methods. The repository is designed to be modular and easy to use, allowing for quick experimentation and please use it with caution for production purposes.

## Installation

```bash
pip install -U multi-agent-llm
```
## Quick example

```python
import os; os.environ['OPENAI_API_KEY'] = "your_openai_api_key"

from multi_agent_llm import OpenAILLM, AGOT
from pydantic import BaseModel, Field

llm = OpenAILLM(model_name="gpt-4o-mini", temperature=0.3)
# Currently we only have wrapper for OpenAI, but it can be easily extended to other LLMs

# Define the answer schema
class QueryAnswer(BaseModel):
    explanation: str = Field(description="Explanation of the answer.")
    answer: str = Field(description="Final multiple-choice answer.")
    answer_label: str = Field(description="Label of the answer. Either A, B, C, or D.")

# Initialize AIOT with the LLM and run a sample query
agot = AGOT(
    llm=llm,
    max_depth=1,
    max_num_layers=3,
    max_new_tasks=3,
    max_concurrent_tasks=10,
)

question = """
Observations of a quasar across the electromagnetic spectrum are being carried out.
Focusing on the near infrared and optical spectrum, there is a peak at a wavelength of about 790 nm,
and at shorter wavelengths < 790 nm the flux drops significantly.

If we lived in a universe defined by a Lambda-CDM model such that the current Hubble constant is 70 km / s / Mpc,
the matter density parameter is 0.3, the dark energy density parameter is 0.7, and the universe is flat,
what can be assumed about the value of the comoving distance (for scale factor a=1) of this quasar from
the Earth?

A. 6 Gpc
B. 7 Gpc
C. 8 Gpc
D. 9 Gpc
"""

response = await agot.run_async(question, schema=QueryAnswer)

answer = response.final_answer

print(f"ANSWER: {answer.answer_label} - {answer.answer}\n")
print(f"EXPLANATION: {answer.explanation}")
```

```
ANSWER: C - 8 Gpc

EXPLANATION: The analysis of the quasar's distance was based on comoving distance calculations employing the Lambda-CDM model parameters. Given the quasar's peak emission wavelength at 790 nm and estimated redshift effects from existing literature, the comoving distance is derived using integration over the redshift, leading to a calculated distance consistent with values observed in other quasars. Previous studies have established that quasars with similar characteristics typically lie at distances ranging from 6 to 9 Gpc. The comprehensive review of analyses culminated in a well-supported final distance assessment placing the quasar at approximately 8 Gpc.
```

## Implemented Methods

| **Method** | **Description**                                                                                                                                                                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AIoT**   | Autonomous Iteration of Thought (AIoT) dynamically adapts its reasoning paths based on the evolving state of the conversation without generating alternate explorative thoughts that are ultimately discarded. [Quick Example](./examples/iot/iot-quick-example.ipynb) |
| **GIoT**   | Guided Iteration of Thought (GIoT) forces the LLM to continue iterating for a predefined number of steps, ensuring a thorough exploration of reasoning paths. [Quick Example](./examples/iot/iot-quick-example.ipynb)                                                  |
| **AGoT**   | Adaptive Graph of Thoughts (AGoT) dynamically constructs a directed acyclic graph (DAG) using layer-wise strategies and recursive self-application to decompose conceptual complexity. [Quick Example](./examples/agot/agot-quick-example.ipynb) |



------

This repository also contains the results for the paper [Iteration of Thought](https://arxiv.org/abs/2409.12618) and *Adaptive Graph of Thoughts* (coming soon). You can find the relevant [experimental setups, datasets, and results](./examples). The folder contains results from various tasks. Feel free to explore these folders to reproduce the experiments or to get a deeper understanding of how the AIOT, GIOT, and AGoT frameworks work in practice.
