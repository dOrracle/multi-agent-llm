import asyncio
import json
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from multi_agent_llm import AGOT, OpenAILLM


async def morehopqa(
    n: int = 100,
    batch_size: int = 50,
    max_num_layers: int = 3,
    max_depth: int = 1,
    max_new_tasks: int = 3,
    verbose: bool = False,
    max_concurrent_tasks: int = 3000
):

    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        temperature=0.3,
        api_key="...",
    )

    multihop_df = pd.read_csv('MoreHopQA.csv')
    np.random.seed(42)
    multihop_df = multihop_df.sample(n=n)

    class QA(BaseModel):
        explanation: str = Field(
            ...,
            description="Explanation for the answer in detail",
        )
        answer: str = Field(
            ...,
            description="Final answer derived from the explanation",
        )
    morehop_agent = AGOT(
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
            context = [multihop_df.iloc[i]['context']]
            question = multihop_df.iloc[i]['question']
            context.append([question + "\n Answer: "])
            task = ' '.join([f'{c}' for c in context])
            tasks.append(morehop_agent.run_async(task, schema=QA))
        outputs = await asyncio.gather(*tasks)
        results.extend(outputs)

    # save results to json
    with open(f"./results/multihop_results_reasoners_n_{n}_batch_{batch_size}_{max_concurrent_tasks}_concurrent_nesting_{max_depth}_layers_{max_num_layers}__nodes_{max_new_tasks}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json", "w") as json_file:
        json.dump([result.dict() for result in results], json_file, indent=4)


async def main():

    tasks = []
    tasks.append(
        morehopqa(
            n=100,
            batch_size=20,
            max_depth=1,
            max_num_layers=3,
            max_new_tasks=3,
            verbose=1,
            max_concurrent_tasks=1000
        )
    )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
