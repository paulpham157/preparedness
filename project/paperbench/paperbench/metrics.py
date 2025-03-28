import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Hashable

import dateutil.parser
import numpy as np
from paperbench.judge.judge import GradedTaskNode, disqualify
from tqdm import tqdm

EXPECTED_PAPERS = 20


@dataclass
class PaperEvaluation:
    """A single evaluation of a paper"""

    paper_run_id: str
    paper_id: str
    graded_task_node: GradedTaskNode


@dataclass
class EvaluationRun:
    """A single run of the Evaluation, i.e. where all papers have been evaluated"""

    seed: Hashable
    paper_evaluations: list[PaperEvaluation]

    def is_complete(self) -> bool:
        return len(self.paper_evaluations) == EXPECTED_PAPERS

    def is_valid(self) -> bool:
        paper_ids = [pe.paper_id for pe in self.paper_evaluations]
        return len(paper_ids) == len(set(paper_ids))


@dataclass
class MetricResult:
    mean: float
    std_err: float
    n_runs: int


@dataclass
class ParsedEntry:
    """Contains parsed JSONL entry data"""

    agent_id: str
    paper_id: str
    paper_run_id: str
    timestamp: float
    graded_task_tree: GradedTaskNode


def compute_ars(
    eval_run: EvaluationRun,
    task_node_transform: Callable[[GradedTaskNode], GradedTaskNode] | None = None,
) -> float:
    """
    Computes the average replication score (ARS) for a single evaluation run
    """
    assert (
        eval_run.is_complete()
    ), f"Evaluation run is not complete: less than {EXPECTED_PAPERS} papers have been evaluated"

    assert eval_run.is_valid(), "Evaluation run contains duplicate paper evaluations"

    scores = []

    for paper_eval in eval_run.paper_evaluations:
        graded_task_node = paper_eval.graded_task_node
        if task_node_transform is not None:
            graded_task_node = task_node_transform(graded_task_node)
        score = graded_task_node.score
        scores.append(score)

    return np.mean(scores).item()


def compute_agg_stats(
    evaluation_runs: list[EvaluationRun],
    compute_ars_kwargs: dict | None = None,
) -> MetricResult:
    """
    Computes aggregate statistics for replication scores across multiple eval runs.
    Returns the mean score, standard error, and number of valid seeds.
    """
    # Filter for complete evaluations (i.e. all papers have been evaluated)
    if compute_ars_kwargs is None:
        compute_ars_kwargs = {}
    complete_evaluations = [eval for eval in evaluation_runs if eval.is_complete()]

    scores = [compute_ars(eval_run, **compute_ars_kwargs) for eval_run in complete_evaluations]

    return MetricResult(
        mean=np.mean(scores).item(),
        std_err=(np.std(scores) / np.sqrt(len(scores))).item() if scores else 0.0,
        n_runs=len(complete_evaluations),
    )


def check_disqualification(
    paper_eval: PaperEvaluation, disquailified_paper_runs: set[str]
) -> PaperEvaluation:
    """
    Checks if a PaperEvaluation is from a disqualified paper run
    and if so sets the graded task node score to 0.0
    """
    if paper_eval.paper_run_id in disquailified_paper_runs:
        disqualified_graded_node = disqualify(paper_eval.graded_task_node)
        paper_eval.graded_task_node = disqualified_graded_node
    return paper_eval


def parse_disqualified_runs(disqualification_data_path: Path) -> set[str]:
    with open(disqualification_data_path, "r") as f:
        return {line.strip() for line in f}


def parse_run_data(
    run_data_path: Path,
    disqualification_data_path: Path | None = None,
    seeds_to_keep: int | None = None,
) -> dict[str, list[EvaluationRun]]:
    """
    Parses run data from JSONL files and organizes it into EvaluationRun objects.
    Keeps only the N most recent seeds for each agent based on timestamps.

    Args:
        run_data_path: Directory path containing evaluation JSONL files
        disqualification_data_path: (Optional) Path to a file where each line is a disqualified
        paper run ID
        seeds_to_keep: (Optional) Number of most recent seeds to keep per agent

    Returns:
        Dictionary mapping agent IDs to lists of EvaluationRun objects
        where each EvaluationRun contains 1 seed of paper evaluations
    """
    agent_runs = {}

    # Helper function to validate and extract required data
    def parse_jsonl_entry(entry: dict) -> ParsedEntry | None:
        if not all(
            [
                entry.get("record_type") == "extra",
                entry.get("data", {}).get("pb_result", {}).get("grader_output"),
                entry.get("data", {}).get("run_group_id"),
                entry.get("timestamp"),
            ]
        ):
            return None

        pb_result = entry["data"]["pb_result"]
        if not pb_result.get("grader_success"):
            return None

        run_group_id = entry["data"]["run_group_id"]
        paper_run_id = entry["data"]["run_id"]
        agent_id = run_group_id.split("_")[-1]
        paper_id = pb_result["paper_id"]
        timestamp = dateutil.parser.parse(entry["timestamp"]).timestamp()
        graded_task_tree = GradedTaskNode.from_dict(pb_result["grader_output"]["graded_task_tree"])

        return ParsedEntry(
            agent_id=agent_id,
            paper_id=paper_id,
            paper_run_id=paper_run_id,
            timestamp=timestamp,
            graded_task_tree=graded_task_tree,
        )

    if disqualification_data_path is not None:
        disqualified_paper_runs = parse_disqualified_runs(disqualification_data_path)
    else:
        disqualified_paper_runs = set()

    for file in tqdm(sorted((run_data_path.glob("*.jsonl"))), desc="Parsing run data"):
        with open(file, "r") as f:
            for line in f:
                entry = json.loads(line)
                parsed_entry = parse_jsonl_entry(entry)
                if not parsed_entry:
                    continue

                # Initialize agent and seed data structures if needed
                if parsed_entry.agent_id not in agent_runs:
                    agent_runs[parsed_entry.agent_id] = {}

                paper_eval = PaperEvaluation(
                    paper_run_id=parsed_entry.paper_run_id,
                    paper_id=parsed_entry.paper_id,
                    graded_task_node=parsed_entry.graded_task_tree,
                )
                paper_eval = check_disqualification(paper_eval, disqualified_paper_runs)

                if parsed_entry.paper_id not in agent_runs[parsed_entry.agent_id]:
                    agent_runs[parsed_entry.agent_id][parsed_entry.paper_id] = []

                agent_runs[parsed_entry.agent_id][parsed_entry.paper_id].append(
                    {"paper_eval": paper_eval, "timestamp": parsed_entry.timestamp}
                )

    # Convert to final format, keeping only the N most recent seeds
    agent_to_eval_runs = {agent: [] for agent in agent_runs.keys()}

    for agent, paper_data in agent_runs.items():

        # keep only N most recent seeds
        filtered_paper_data = {}
        for paper_id, data in paper_data.items():
            sorted_data = sorted(data, key=lambda x: x["timestamp"], reverse=True)
            filtered_paper_data[paper_id] = sorted_data[:seeds_to_keep]

        # then, create the N EvaluationRun objects
        max_num_seeds = max([len(data) for data in filtered_paper_data.values()])
        for seed in range(max_num_seeds):
            eval_run = EvaluationRun(seed=seed, paper_evaluations=[])
            for paper_id, data in filtered_paper_data.items():
                # some evaluation runs may not have all papers evaluated
                if seed == len(data):
                    continue
                paper_eval = data[seed]["paper_eval"]
                eval_run.paper_evaluations.append(paper_eval)

            agent_to_eval_runs[agent].append(eval_run)

    return agent_to_eval_runs


if __name__ == "__main__":
    """Example usage of parse_run_data and compute_agg_stats"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-path", type=Path, required=True)
    parser.add_argument(
        "--disqualified-runs-path",
        type=Path,
        required=False,
        default=None,
        help="Path to file where each line is a disqualified paper run ID",
    )
    args = parser.parse_args()

    agent_to_eval_runs = parse_run_data(
        args.run_data_path, args.disqualified_runs_path, seeds_to_keep=3
    )

    results = {}
    for agent in agent_to_eval_runs.keys():
        results[agent] = compute_agg_stats(agent_to_eval_runs[agent])

    results = {agent: asdict(metric) for agent, metric in results.items()}

    import pandas as pd

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("mean", ascending=False)

    print(results_df)
