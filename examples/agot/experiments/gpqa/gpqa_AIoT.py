import asyncio
import csv
import json
import random
import warnings
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from rich import print
from tqdm.asyncio import tqdm

from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.iteration_of_thought import AIOT


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


async def process_question(aiot: AIOT, task, item: GPQAQuestion):
    result = await aiot.run_async(task(item))
    try:
        print(
            f"{result.answer.answer}, {item.correct_answer}, "
            f"{result.answer.answer.strip().upper() == item.correct_answer}"
        )
        return [result.answer, result.thoughts]
    except Exception as e:
        print(f"Error processing question: {item.question}")
        # raise error
        raise e


async def process_batch(aiot: AIOT, task, batch: List[GPQAQuestion]):
    results = await asyncio.gather(*[process_question(aiot, task, item) for item in batch])
    final_results = [result[0] for result in results]
    final_thoughts = [result[1] for result in results]
    return final_results, final_thoughts


async def main(
    file_path: str,
    answer_schema: BaseModel,
    n: int = 100,
    batch_size: int = 10,
    iterations: int = 5,
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

    # Initialize AIOT
    aiot = AIOT(
        llm=OpenAILLM(
            api_key="...",
            model_name="gpt-4o-mini",
            temperature=0.3
        ),
        iterations=iterations,
        answer_schema=answer_schema,
    )

    # Process questions in batches with progress bar
    print("Processing questions...")
    results = []
    thought = []
    for i in tqdm(range(0, n, batch_size)):
        batch = test_data[i: i + batch_size]
        batch_results, batch_thoughts = await process_batch(aiot, task, batch)
        results.extend(batch_results)
        thought.extend(batch_thoughts)

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
    # Save results to JSON
    save_results_to_json(
        results,
        f"./results/gpqa_AIOT_n_{n}_batch_{batch_size}_{iterations}_iterations_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
    )
    save_results_to_json(
        thought,
        f"./results/gpqa_AIOT_thoughts_n_{n}_batch_{batch_size}_{iterations}_iterations_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
    )

    # Use max_concurrent_tasks in the filename (use different value)
    # Save the graph object

if __name__ == "__main__":
    # Replace async with threads/processes
    import time
    start = time.time()
    # Run the main function with asyncio
    asyncio.run(
        main(
            "./gpqa_diamond.csv",
            n=198,
            batch_size=100,
            iterations=10,
            answer_schema=QA,
        )
    )
    print(f"Time taken: {time.time()-start}")
