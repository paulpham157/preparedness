import chz
from nanoeval.eval import EvalSpec, RunnerArgs
from nanoeval.evaluation import run
from nanoeval.setup import nanoeval_entrypoint
from paperbench.nano.eval import PaperBench
from paperbench.nano.utils import run_sanity_checks


@chz.chz
class DefaultRunnerArgs(RunnerArgs):
    concurrency: int = 5


async def main(paperbench: PaperBench, runner: DefaultRunnerArgs) -> None:
    run_sanity_checks(paperbench)
    await run(EvalSpec(eval=paperbench, runner=runner))


if __name__ == "__main__":
    nanoeval_entrypoint(chz.entrypoint(main))
