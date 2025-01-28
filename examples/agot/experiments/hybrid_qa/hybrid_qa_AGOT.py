import asyncio
import json
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from multi_agent_llm import AGOT, OpenAILLM


async def hybridqa(
    n: int = 100,
    batch_size: int = 50,
    max_num_layers: int = 3,
    max_depth: int = 1,
    max_new_tasks: int = 3,
    verbose: bool = False,
    max_concurrent_tasks: int = 3000
):

    llm = OpenAILLM(
        api_key="...",
        model_name="gpt-4o-mini",
        temperature=0.3
    )

    hybrid_qa = load_dataset("wenhu/hybrid_qa", trust_remote_code=True)
    hybrid_qa = hybrid_qa['train']
    # Make it into a dataframe
    hybrid_qa_df = pd.DataFrame(hybrid_qa)
    np.random.seed(0)
    hybrid_qa_df = hybrid_qa_df.sample(n=n)

    class QA(BaseModel):
        explanation: str = Field(
            ...,
            description="Explanation for the answer in detail",
        )
        answer: str = Field(
            ...,
            description="Final answer derived from the explanation",
        )

    hybrid_agent = AGOT(
        llm=llm,
        max_depth=max_depth,
        max_num_layers=max_num_layers,
        max_new_tasks=max_new_tasks,
        verbose=verbose,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    results = []
    for j in tqdm(range(0, n, batch_size)):
        tasks = []
        for i in range(j, min(j+batch_size, n)):
            context = hybrid_qa_df.iloc[i]['table']
            context = {k: v for k, v in context.items() if k not in [
                'uid', 'intro', 'section_title', 'section_text']}
            table = ' '.join([f'{k}: {v}' for k, v in context.items()])
            question = hybrid_qa_df.iloc[i]['question']
            # Join the context dictionary and question to form the task
            task = table + f'\n Question: {question}' + ' Answer: '
            task = "Using the context table, answer the question without any explanation. \n Table:" + task
            tasks.append(hybrid_agent.run_async(task, schema=QA))
        outputs = await asyncio.gather(*tasks)
        results.extend(outputs)

    # save results to json
    with open(f"./results/hybridqa_results_reasoners_n_{n}_batch_{batch_size}_{max_concurrent_tasks}_concurrent_nesting_{max_depth}_layers_{max_num_layers}__nodes_{max_new_tasks}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json", "w") as json_file:
        json.dump([result.dict() for result in results], json_file, indent=4)


async def main():

    tasks = []
    tasks.append(
        hybridqa(n=100, batch_size=10, max_depth=1,
                 max_num_layers=3,
                 max_new_tasks=3,
                 verbose=1,
                 max_concurrent_tasks=1000)
    )
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
