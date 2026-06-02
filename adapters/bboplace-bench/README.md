# BBOPlace-Bench -> Harbor Adapter

This adapter generates Harbor-style tasks for BBOPlace-Bench. Each task wraps a
single `(benchmark, placer)` pair and exposes an iterative black-box evaluator
to the agent.

The agent writes `/app/solution.py` and can run:

```bash
python3 /app/submit.py --info
bash /app/submit.sh
```

`solution.py` may define `solve(info)`, `generate(info)`, `CANDIDATES`, or
`CANDIDATE`. The value should be one candidate vector or a list of candidate
vectors. The evaluator minimizes HPWL and reports Harbor reward as:

```text
reward = 1 / (1 + hpwl / 1e5)
```

The final verifier evaluates `/app/solution.py` and compares it with the best
successful iterative submission recorded in `/logs/agent/submissions.jsonl`.
The reported reward is the better of those two.

## Benchmark Data

BBOPlace-Bench requires external benchmark data before a runnable Harbor task
can be generated. The benchmark files are not committed to Frontier-CS and are
not downloaded by the agent during a trial.

Download the original ISPD2005 and ICCAD2015 benchmark datasets linked from
`bboplace/README.md`, extract them, and place them under:

```text
bboplace/benchmarks/
  ispd2005/
    adaptec1/
    adaptec2/
    ...
  iccad2015/
    superblue1/
    superblue3/
    ...
```

The generator copies only the selected benchmark into the Harbor judge image.
The agent workspace still does not contain the benchmark files; the agent gets
metadata through `python3 /app/submit.py --info` and score feedback through
`bash /app/submit.sh`.

## Frontier-CS Wrapper

For normal Frontier-CS usage, run the wrapper directly. Like the Algorithmic
and 2.0 Harbor tracks, it auto-generates the missing Harbor task before the
trial starts:

```bash
uv run frontier harbor trial bboplace adaptec1 -a codex -m gpt-5.5 --json
```

Use explicit generation only when you want to pre-generate a task or refresh an
existing generated task after adding or changing benchmark data:

```bash
uv run frontier harbor generate bboplace adaptec1 --overwrite
```

## Generate Tasks

The lower-level adapter CLI is useful when running Harbor directly instead of
the Frontier-CS wrapper. From the repository root:

```bash
PYTHONPATH=adapters/bboplace-bench/src \
python3 -m bboplace_bench_harbor.main \
  --source bboplace \
  --output-dir datasets/bboplace-bench \
  --benchmarks adaptec1 \
  --placers mgo \
  --overwrite
```

Generate tasks only after the needed benchmark directory exists. The generator
copies the selected benchmark into the Harbor judge image.

If you are only inspecting the generated Harbor shape before downloading data,
add:

```bash
--allow-missing-benchmark
```

Tasks generated without benchmark data are structural only and will not run
until regenerated with the data present.

## Run with Harbor

```bash
uv run harbor trial start -p datasets/bboplace-bench/bboplace-bench-mgo-adaptec1-mp
```

The generated task uses two services:

- `main`: the agent workspace with `/app/submit.sh`, `/app/submit.py`, and
  static task metadata only. It does not contain the BBOPlace repository,
  benchmark data, or evaluator runtime.
- `judge`: a black-box HTTP evaluator that owns the BBOPlace repository and
  benchmark data. The agent submits candidate vectors to this service; the
  judge does not execute agent Python code.

The final verifier reads judge-recorded submissions from the judge HTTP API and
uses the best successful iterative submission. It does not execute
`/app/solution.py` inside an environment that contains the BBOPlace repository
or benchmark data.

## Current Scope

The default supported mode is `eval_gp_hpwl=False` with `mgo` or `sp`. `hpo` and
global-placement HPWL import DREAMPlace at runtime, so they require a
DREAMPlace-enabled Docker image and local benchmark data.
