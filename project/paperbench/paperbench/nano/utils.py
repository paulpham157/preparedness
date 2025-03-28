import logging
from datetime import datetime, timedelta
from pathlib import Path

import structlog
from alcatraz.clusters.local import LocalConfig
from paperbench.constants import AGENT_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.utils import get_default_runs_dir


def uses_local_config(paperbench: "PaperBench") -> bool:
    """
    Check if any of paperbench.solver.cluster_config, paperbench.reproduction.cluster_config,
    or paperbench.judge.cluster_config is an instance of LocalConfig.

    Args:
        paperbench: A PaperBench PythonCodingEval instance

    Returns:
        bool: True if any of the cluster configs is a LocalConfig, False otherwise
    """

    # PythonCodingSolver may not have a cluster_config, just ExternalPythonCodingSolver does for now
    if hasattr(paperbench.solver, "cluster_config"):
        if isinstance(paperbench.solver.cluster_config, LocalConfig):
            return True

    # Check reproduction's cluster_config
    if isinstance(paperbench.reproduction.cluster_config, LocalConfig):
        return True

    # Check judge's cluster_config
    if isinstance(paperbench.judge.cluster_config, LocalConfig):
        return True

    return False


def build_agent_command(agent: "Agent") -> str:
    """Builds the command to run the agent."""

    cmd = ["bash", f"{AGENT_DIR}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    return " ".join(cmd)


def build_reproduce_command(task: "PaperBenchTask") -> str:
    """Builds the command to run the reproduction."""

    cmd = [
        f"python3 {WORKSPACE_BASE}/run_reproduce.py",
        f"--submission-path {SUBMISSION_DIR}",
        "--out-path /output/reproduction_metadata.json",
    ]

    if task.reproduction.timeout:
        cmd.extend(["--timeout", str(task.reproduction.timeout)])

    return " ".join(map(str, cmd))


def build_judge_command(judge: "JudgeConfig", task: "PaperBenchTask") -> str:
    """Builds the command to run the judge."""

    cmd = [
        "/opt/conda/envs/grader/bin/python",  # Use the conda env installed in `pb-grader`
        f"{WORKSPACE_BASE}/run_judge.py",  # Assumes judge script exists here
        f"--submission-path {SUBMISSION_DIR}",
        f"--paper-id {task.paper_id}",
        f"--judge {judge.scaffold}",
        "--out-dir /output",
    ]

    if judge.model not in ("dummy", "random"):
        cmd.extend(["--model", judge.model])

    if judge.reasoning_effort:
        cmd.extend(["--reasoning-effort", judge.reasoning_effort])

    if judge.code_only:
        cmd.append("--code-only")

    if judge.max_depth:
        cmd.extend(["--max-depth", str(judge.max_depth)])

    return " ".join(map(str, cmd))


def get_file_at_duration(
    files: list[str], duration_hr: int, logger: logging.Logger
) -> tuple[str, timedelta]:
    """
    Given a list of files with timestamped names, return the file closest to `duration_hr`-hours
    after the earliest file in the list.
    e.g.
    ```
    files = [
        "path/to/file/2024-12-07T10-19-52-GMT.tar.gz",
        "path/to/file/2024-12-07T10-49-55-GMT.tar.gz",
        "path/to/file/2024-12-07T11-19-56-GMT.tar.gz",
        "path/to/file/2024-12-07T11-49-56-GMT_step_10.tar.gz",
        "path/to/file/2024-12-07T12-19-58-GMT.tar.gz",
    ]
    get_file_at_duration(files, 1)
    > "path/to/file/2024-12-07T11-19-56-GMT.tar.gz",
    ```
    """
    # Extract timestamps from filenames
    timestamps = []
    for file in files:
        # Extract timestamp string between last / and .tar.gz
        ts_str = file.split("/")[-1].replace(".tar.gz", "")
        if "step" in ts_str:
            ts_str = ts_str.split("_step_")[0]
        # Parse timestamp string into datetime
        try:
            # Try parsing with timezone
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S-%Z")
        except ValueError:
            # Fallback to GMT if no timezone specified
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S-GMT")
        timestamps.append(dt)

    earliest = min(timestamps)
    target = earliest + timedelta(hours=duration_hr)

    # Find file with timestamp closest to target
    closest_file = min(zip(files, timestamps), key=lambda x: abs((x[1] - target).total_seconds()))
    logger.info(
        f"Closest file to {duration_hr} hours after earliest file ({earliest}) is {closest_file[0]}"
    )
    retrieved_file = closest_file[0]
    retrieved_duration = closest_file[1] - earliest
    return retrieved_file, retrieved_duration


def file_processor(logger, method_name, original_event_dict):
    event_dict = original_event_dict.copy()  # Avoid mutating the original

    destinations = event_dict.pop("destinations", [])
    run_group_id = event_dict.pop("run_group_id", None)
    run_id = event_dict.pop("run_id", None)
    runs_dir = event_dict.pop("runs_dir", get_default_runs_dir())

    if "run" in destinations and run_group_id and run_id:
        dst = Path(runs_dir) / run_group_id / run_id / "run.log"
        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "a") as f:
            f.write(str(event_dict) + "\n")

    if "group" in destinations and run_group_id:
        dst = Path(runs_dir) / run_group_id / "group.log"
        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "a") as f:
            f.write(str(event_dict) + "\n")

    return original_event_dict


def filter_processor(logger, method_name, event_dict):
    destinations = event_dict.pop("destinations", [])
    event_dict.pop("run_group_id", None)
    event_dict.pop("run_id", None)
    event_dict.pop("runs_dir", None)

    if "console" not in destinations:
        raise structlog.DropEvent()

    return event_dict
