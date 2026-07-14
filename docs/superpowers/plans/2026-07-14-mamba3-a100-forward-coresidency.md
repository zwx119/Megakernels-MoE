# Mamba3 A100 Forward Co-residency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether reducing the forward Mamba3 SISO kernel's warp and register configuration enables useful two-stream co-residency on A100 at sequence length 16384.

**Architecture:** Keep the isolated Mamba3 vendor copy and mathematical split unchanged. Select one Triton launch configuration per process through validated environment variables, pre-screen six configurations with CUDA event timing, then use Nsight Systems to measure the promising configurations and compare them against the fixed `8 warps / maxnreg=256 / 1 stage` FULL baseline.

**Tech Stack:** Python 3.11, PyTorch CUDA events and streams, Triton 3.5, Nsight Systems, shell, `unittest`.

## Global Constraints

- GPU is one A100-SXM4-80GB.
- Shape is BS=1, hidden size=2560, state size=128, expand=2, head dim=64, 80 heads, chunk size=64, BF16.
- Sequence length is 16384.
- Scan `num_warps` in `{4, 8}`, `maxnreg` in `{128, 192, 256}`, and keep `num_stages=1`.
- The baseline is always FULL with `8 warps / maxnreg=256 / 1 stage`.
- Continue beyond 16K only if core speedup is at least 1.03x and complete block time does not regress.
- Do not modify backward, add a fused merge, or extrapolate the A100 result to H100.

---

### Task 1: Make the isolated forward kernel launch configuration explicit

**Files:**
- Modify: `experiments/mamba3_h_overlap_0714/vendor/mamba3_siso_fwd.py`
- Modify: `tests/test_mamba3_h_ablation.py`

**Interfaces:**
- Consumes: process environment variables `MAMBA3_FWD_NUM_WARPS`, `MAMBA3_FWD_MAXNREG`, and `MAMBA3_FWD_NUM_STAGES`.
- Produces: `_env_choice(name: str, default: int, allowed: tuple[int, ...]) -> int` and one deterministic `triton.Config` selected at import time.

- [ ] **Step 1: Write the failing source-contract test**

```python
def test_forward_launch_config_is_selectable_from_environment(self) -> None:
    source = FWD_SOURCE.read_text()
    self.assertIn("MAMBA3_FWD_NUM_WARPS", source)
    self.assertIn("MAMBA3_FWD_MAXNREG", source)
    self.assertIn("MAMBA3_FWD_NUM_STAGES", source)
    self.assertIn("_env_choice", source)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python3 -m unittest tests.test_mamba3_h_ablation.Mamba3HAblationSourceTests.test_forward_launch_config_is_selectable_from_environment -v`

Expected: FAIL because the environment-variable names are absent.

- [ ] **Step 3: Add validated import-time configuration**

Add near the forward module constants:

```python
def _env_choice(name: str, default: int, allowed: tuple[int, ...]) -> int:
    value = int(os.environ.get(name, default))
    if value not in allowed:
        choices = ", ".join(str(choice) for choice in allowed)
        raise ValueError(f"{name} must be one of {choices}, got {value}")
    return value


_FWD_NUM_WARPS = _env_choice("MAMBA3_FWD_NUM_WARPS", 8, (4, 8))
_FWD_MAXNREG = _env_choice("MAMBA3_FWD_MAXNREG", 256, (128, 192, 256))
_FWD_NUM_STAGES = _env_choice("MAMBA3_FWD_NUM_STAGES", 1, (1,))
```

Replace the fixed autotune entry with:

```python
configs=[triton.Config(
    {},
    num_stages=_FWD_NUM_STAGES,
    num_warps=_FWD_NUM_WARPS,
    maxnreg=_FWD_MAXNREG,
)],
```

- [ ] **Step 4: Run all source-contract tests**

Run: `python3 -m unittest tests.test_mamba3_h_ablation -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the configurable launch change**

```bash
git add experiments/mamba3_h_overlap_0714/vendor/mamba3_siso_fwd.py tests/test_mamba3_h_ablation.py
git commit -m "feat: parameterize Mamba3 forward launch resources"
```

### Task 2: Add deterministic event pre-screening and Nsight sweep orchestration

**Files:**
- Create: `experiments/mamba3_h_overlap_0714/bench_fwd_coresidency.py`
- Create: `experiments/mamba3_h_overlap_0714/run_fwd_coresidency_sweep.sh`
- Modify: `tests/test_mamba3_h_ablation.py`

**Interfaces:**
- Consumes: the isolated `MAMBA3_DIR`, one H mode, and one launch configuration per process.
- Produces: JSON event records and profiler files named `fwd_seq16384_<mode>_w<warps>_nr<maxnreg>_s1_rep<repeat>`.

- [ ] **Step 1: Add failing runner-contract tests**

```python
RUNNER = ROOT / "experiments/mamba3_h_overlap_0714/run_fwd_coresidency_sweep.sh"
BENCH = ROOT / "experiments/mamba3_h_overlap_0714/bench_fwd_coresidency.py"

def test_forward_coresidency_runner_carries_config_into_each_process(self) -> None:
    runner = RUNNER.read_text()
    self.assertIn("MAMBA3_FWD_NUM_WARPS", runner)
    self.assertIn("MAMBA3_FWD_MAXNREG", runner)
    self.assertIn("WARPS_LIST", runner)
    self.assertIn("MAXNREG_LIST", runner)

def test_forward_coresidency_benchmark_uses_cuda_events(self) -> None:
    source = BENCH.read_text()
    self.assertIn("torch.cuda.Event", source)
    self.assertIn("elapsed_time", source)
```

- [ ] **Step 2: Verify the runner tests fail**

Run: `python3 -m unittest tests.test_mamba3_h_ablation -v`

Expected: FAIL because the benchmark and runner files do not exist.

- [ ] **Step 3: Implement the event benchmark**

Create a CLI that accepts `--mamba-dir`, `--seq`, `--warmup`, `--iters`, and `--output`; constructs the same SISO Mamba3 shape as `dump_full_reference.py`; performs forward-only warmup; records one synchronized CUDA event interval per iteration; and writes:

```json
{
  "seq": 16384,
  "mode": "overlap",
  "num_warps": 4,
  "maxnreg": 128,
  "num_stages": 1,
  "median_ms": 0.0,
  "samples_ms": []
}
```

The values for mode and launch resources come from the corresponding environment variables. Use a fixed seed and `torch.inference_mode()` so configurations receive identical inputs.

- [ ] **Step 4: Implement the resumable shell runner**

The runner must:

```bash
SEQ=${SEQ:-16384}
WARPS_LIST=${WARPS_LIST:-4,8}
MAXNREG_LIST=${MAXNREG_LIST:-128,192,256}
NUM_STAGES=${NUM_STAGES:-1}
MODES=${MODES:-no_h,h_only,overlap}
REPEATS=${REPEATS:-1}
```

Run the fixed FULL baseline only as `w8_nr256_s1`; run event timing for every requested mode/configuration; then run Nsight for every requested mode/configuration/repeat. Export the three launch variables and mode for each child process. Skip a child only when its final JSON or parsed `.overall.csv` exists. Reuse persistent Triton and TorchInductor cache directories keyed by `w<warps>_nr<maxnreg>_s<stages>`.

- [ ] **Step 5: Validate syntax and source contracts**

Run:

```bash
python3 -m py_compile experiments/mamba3_h_overlap_0714/bench_fwd_coresidency.py
bash -n experiments/mamba3_h_overlap_0714/run_fwd_coresidency_sweep.sh
python3 -m unittest tests.test_mamba3_h_ablation -v
```

Expected: all commands succeed and tests PASS.

- [ ] **Step 6: Commit the experiment runner**

```bash
git add experiments/mamba3_h_overlap_0714/bench_fwd_coresidency.py experiments/mamba3_h_overlap_0714/run_fwd_coresidency_sweep.sh tests/test_mamba3_h_ablation.py
git commit -m "feat: add Mamba3 forward co-residency sweep"
```

### Task 3: Add a baseline-stable co-residency summarizer

**Files:**
- Create: `experiments/mamba3_h_overlap_0714/summarize_fwd_coresidency.py`
- Create: `tests/test_mamba3_fwd_coresidency_summary.py`

**Interfaces:**
- Consumes: profiler `.overall.csv` files under `<root>/parsed` using the runner naming contract.
- Produces: `<root>/fwd_coresidency_summary.csv` and `<root>/fwd_coresidency_summary.md`.

- [ ] **Step 1: Write a failing median-and-baseline unit test**

Create three synthetic repeats for fixed FULL `w8_nr256_s1` and for one `no_h`, `h_only`, and `overlap` configuration. Invoke `summarize(root)` and assert:

```python
self.assertAlmostEqual(row["concurrent_ms"], row["overlap_core_sum_ms"] - row["overlap_core_busy_ms"])
self.assertAlmostEqual(row["core_speedup"], baseline_core_busy / row["overlap_core_busy_ms"])
self.assertAlmostEqual(row["block_speedup"], baseline_block_span / row["overlap_block_span_ms"])
self.assertEqual(row["passes_gate"], row["core_speedup"] >= 1.03 and row["block_speedup"] >= 1.0)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python3 -m unittest tests.test_mamba3_fwd_coresidency_summary -v`

Expected: FAIL because `summarize_fwd_coresidency` does not exist.

- [ ] **Step 3: Implement strict filename parsing and median aggregation**

Use this filename contract:

```python
NAME_RE = re.compile(
    r"^fwd_seq(?P<seq>\d+)_(?P<mode>full|no_h|h_only|overlap)_"
    r"w(?P<warps>\d+)_nr(?P<maxnreg>\d+)_s(?P<stages>\d+)_"
    r"rep(?P<repeat>\d+)\.overall\.csv$"
)
```

Aggregate `block_span_ms`, `core_sum_ms`, `core_busy_ms`, and `core_span_ms` with the median. Always obtain baseline metrics from `(mode=full, warps=8, maxnreg=256, stages=1)`. Emit one row per complete `no_h`/`h_only`/`overlap` configuration with independent branch latency, concurrent time, speedups, and `passes_gate`.

- [ ] **Step 4: Run summary and regression tests**

Run:

```bash
python3 -m unittest tests.test_mamba3_fwd_coresidency_summary -v
python3 -m unittest tests.test_mamba3_h_ablation -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the summarizer**

```bash
git add experiments/mamba3_h_overlap_0714/summarize_fwd_coresidency.py tests/test_mamba3_fwd_coresidency_summary.py
git commit -m "feat: summarize Mamba3 forward co-residency"
```

### Task 4: Execute the A100 experiment and apply the continuation gate

**Files:**
- Synchronize: `experiments/mamba3_h_overlap_0714/` to the isolated worker experiment directory.
- Generate locally: `experiments/mamba3_h_overlap_0714/results/fwd_coresidency_summary.csv`
- Generate locally: `experiments/mamba3_h_overlap_0714/results/fwd_coresidency_summary.md`

**Interfaces:**
- Consumes: the active one-GPU Merlin worker, existing Mamba3 environment, profile script, collector, and persistent compile caches.
- Produces: repeatable event timing, Nsight reports, parsed metrics, and the 16K gate decision.

- [ ] **Step 1: Confirm worker and environment state**

Verify hostname, GPU model, no competing compute process, free disk space, Python environment, Mamba3 isolated copy, Triton 3.5, and Nsight path before synchronization.

- [ ] **Step 2: Synchronize only experiment files and patch the isolated vendor copy**

Copy the benchmark, runner, summarizer, and modified forward vendor file. Do not alter the canonical Mamba3 checkout or the StateFlow training repository.

- [ ] **Step 3: Run correctness checks**

Generate deterministic official and overlap outputs using `dump_full_reference.py` at a small sequence length for all six launch configurations, then run `compare_full_reference.py`. Expected: every configuration prints `FULL_MATCH_OK` within the existing BF16 tolerances.

- [ ] **Step 4: Run one-repeat 16K pre-screen**

Run the complete `4/8 warps x 128/192/256 maxnreg` matrix with one Nsight repeat. Inspect event times and summarize all six configurations. Expected: six complete rows or explicit compile/runtime failures retained as failures.

- [ ] **Step 5: Repeat promising configurations three times**

Select configurations with increased concurrent time and the lowest overlap `core_busy_ms`; rerun with `REPEATS=3`. Resume logic must retain the first capture and add repeats 2 and 3.

- [ ] **Step 6: Collect and copy results locally**

Run `summarize_fwd_coresidency.py`, copy the CSV and Markdown into the local results directory, and verify every reported median has three samples for the selected configurations.

- [ ] **Step 7: Apply the gate**

If any configuration has `core_speedup >= 1.03` and `block_speedup >= 1.0`, recommend an 8K--64K follow-up sweep. Otherwise stop and report whether resource tuning increased concurrency without improving time, or whether forward remained serialized on A100.

- [ ] **Step 8: Run final local verification**

Run:

```bash
python3 -m unittest tests.test_mamba3_h_ablation tests.test_mamba3_fwd_coresidency_summary -v
python3 -m py_compile experiments/mamba3_h_overlap_0714/bench_fwd_coresidency.py experiments/mamba3_h_overlap_0714/summarize_fwd_coresidency.py
bash -n experiments/mamba3_h_overlap_0714/run_fwd_coresidency_sweep.sh
```

Expected: all tests and syntax checks PASS.
