import asyncio
import csv
import json
import os
import random
import sys
import warnings
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from rich import print
from tqdm.asyncio import tqdm

from multi_agent_llm import AGOT, OpenAILLM


class GPQAQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str


class QA(BaseModel):
    explanation: str = Field(
        ...,
        description="Explanation for the answer"
    )
    answer: str = Field(
        ...,
        description="Final answer (only give A/B/C/D/Uncertain) without any additional explanation.",
    )


class GPQADataset:
    def __init__(self, file_path: str):
        self.questions = self._load_questions(file_path)

    def _load_questions(self, file_path: str) -> List[GPQAQuestion]:
        questions = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                options = [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]
                random.shuffle(options)  # Shuffle the options
                correct_index = options.index(row["Correct Answer"])
                correct_label = ["A", "B", "C", "D"][correct_index]
                question = GPQAQuestion(
                    question=row["Question"],
                    options=options,
                    correct_answer=correct_label,
                )
                questions.append(question)
        return questions

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> GPQAQuestion:
        return self.questions[index]


def save_results_to_json(results: List[BaseModel], filepath: str):
    with open(filepath, "w") as json_file:
        result_list = []
        for result in results:
            try:
                result_list.append(result.dict())
            except:
                result_list.append([node.dict() for node in result])
                continue
        json.dump(result_list, json_file, indent=4)


def calculate_metrics(correct_answers: int, total: int):
    accuracy = correct_answers / total * 100
    return accuracy


async def process_question(agot: AGOT, task, item: GPQAQuestion):
    result = await agot.run_async(task(item), schema=QA)
    try:
        print(
            f"{result.final_answer.answer}, {item.correct_answer}, {result.final_answer.answer.strip().upper() == item.correct_answer}"
        )
        return [result.final_answer, result.graph]
    except Exception as e:
        print(f"Error processing question: {item.question}")
        # raise error
        raise e


async def process_batch(agot: AGOT, task, batch: List[GPQAQuestion]):
    results = await asyncio.gather(*[process_question(agot, task, item) for item in batch])
    final_answers = [result[0] for result in results]
    final_graphs = [result[1] for result in results]
    return final_answers, final_graphs


async def main(
    file_path: str,
    n: int = 100,
    batch_size: int = 10,
    max_num_layers: int = 2,
    max_new_tasks: int = 3,
    max_depth: int = 1,
    verbose: int = 0,
    max_concurrent_tasks: int = 500,
):

    # Load the dataset
    dataset = GPQADataset(file_path)

    # Sample random indices
    random.seed(1993)
    indices = random.sample(range(len(dataset)), n)
    test_data = [dataset[i] for i in indices]

    # Define the task format function
    task = (
        lambda x: f"""Answer the following multiple-choice question. Provide your final answer as 'A', 'B', 'C', or 'D', followed by a brief explanation.

Question: {x.question}

Options:
A. {x.options[0]}
B. {x.options[1]}
C. {x.options[2]}
D. {x.options[3]}

Answer (A/B/C/D)
"""
    )

    # Initialize AGOT
    agot = AGOT(
        llm=OpenAILLM("gpt-4o", temperature=0.3),
        max_depth=max_depth,
        max_num_layers=max_num_layers,
        max_new_tasks=max_new_tasks,
        verbose=verbose,
        max_concurrent_tasks=max_concurrent_tasks,
    )

    # Process questions in batches with progress bar
    print("Processing questions...")
    results = []
    graphs = []
    for i in tqdm(range(0, n, batch_size)):
        batch = test_data[i: i + batch_size]
        try:
            batch_results = await process_batch(agot, task, batch)
            save_results_to_json(
                batch_results[0],
                f"./results/gpqa_AGOT_4o_batch_{i}_{max_concurrent_tasks}_concurrent_nesting_{max_depth}_layers_{max_num_layers}__nodes_{max_new_tasks}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
            )
            save_results_to_json(
                batch_results[1],
                f"./results/graphs_gpqa_AGOT_4o_batch_{i}_{max_concurrent_tasks}_concurrent_nesting_{max_depth}_layers_{max_num_layers}__nodes_{max_new_tasks}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
            )
            results.extend(batch_results[0])
            graphs.extend(batch_results[1])
        except Exception as e:
            print(f"Error processing batch: {i}")
            raise e

    if len(results) != len(test_data):
        warnings.warn("Results and test data are not the same length")
    # Calculate accuracy
    correct_answers = sum(
        1
        for r, q in zip(results, test_data)
        if r.answer.strip().upper() == q.correct_answer
    )

    accuracy = calculate_metrics(correct_answers, len(results))

    print(f"Accuracy: {accuracy:.2f}%")

    # Prepare metadata
    class MetaData(BaseModel):
        indices: List[int]
        accuracy: float
        predictions: List[str]
        correct_answers: List[str]
    predictions = []
    for r in results:
        try:
            predictions.append(r.answer)
        except:
            predictions.append("No Answer")
    metadata = MetaData(
        indices=indices,
        accuracy=accuracy,
        predictions=predictions,
        correct_answers=[q.correct_answer for q in test_data],
    )
    # Add the metadata to the results list
    results.append(metadata)
    graphs.append(metadata)
    # Save results to JSON
    save_results_to_json(
        results,
        f"./results/gpqa_AGOT_4o_n_{n}_batch_{batch_size}_{max_concurrent_tasks}_concurrent_nesting_{max_depth}_layers_{max_num_layers}__nodes_{max_new_tasks}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
    )
    save_results_to_json(
        graphs,
        f"./results/graphs_gpqa_AGOT_4o_n_{n}_batch_{batch_size}_{max_concurrent_tasks}_concurrent_nesting_{max_depth}_layers_{max_num_layers}__nodes_{max_new_tasks}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
    )

if __name__ == "__main__":
    import time
    start = time.time()
    # Run the main function with asyncio
    asyncio.run(
        main(
            "./gpqa_diamond.csv",
            n=198,
            batch_size=35,
            max_depth=1,
            max_num_layers=3,
            max_new_tasks=3,
            verbose=0,
            max_concurrent_tasks=2000,
        )
    )
    print(f"Time taken: {time.time()-start}")
