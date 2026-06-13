# Parallel Gradient Descent — Polynomial Regression via Self-Patching Machine Code

> _"An absolutely criminal way of input-process-output and must not be repeated."_ — the author, in the source code

A two-file C program that trains a degree-8 polynomial to approximate a noisy sine function using multiprocess gradient descent, a live learning-rate scheduler, and a runtime code patcher that rewrites the learning-rate constant **directly in the `.text` (executable) segment** using sentinel-guarded binary search.

---

## Table of Contents

1. [What It Does](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#what-it-does)
2. [Architecture Overview](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#architecture-overview)
3. [Prerequisites & Build](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#prerequisites--build)
4. [Data Format](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#data-format)
5. [Hyperparameters](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#hyperparameters)
6. [Section-by-Section Breakdown](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-by-section-breakdown)
    - [Section 1 — Headers, Macros & Shared Memory](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-1--headers-macros--shared-memory)
    - [Section 2 — Data Loader](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-2--data-loader)
    - [Section 3 — The Runtime Code Patcher](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-3--the-runtime-code-patcher)
    - [Section 4 — Worker Children](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-4--worker-children)
    - [Section 5 — Scheduler Child](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-5--scheduler-child)
    - [Section 6 — `main`](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#section-6--main)
7. [Process Topology](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#process-topology)
8. [The Patching Trick in Detail](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#the-patching-trick-in-detail)
9. [Scheduler Logic](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#scheduler-logic)
10. [Known Quirks & Limitations](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#known-quirks--limitations)
11. [Debugging](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#debugging)
12. [License](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#license)

---

## What It Does

The program fits a polynomial of degree **K = 8** to **N = 100** (x, y) pairs where y ≈ sin(x) + Gaussian noise. Training runs for **5000 epochs** of batch gradient descent.

What makes it unusual:

- **One child process per weight** — K+1 = 9 worker processes are forked every epoch, each computing one partial derivative in parallel, then writing it to shared memory before exiting.
- **A long-lived scheduler process** monitors the loss window and decides when to reduce the learning rate.
- **The learning rate lives in `.text`** — it is embedded as a literal 64-bit IEEE 754 double inside an inline-assembly block, surrounded by two sentinel magic numbers. When the scheduler wants to lower the rate, it asks the parent to locate those sentinels at runtime, `mprotect` the page to `RWX`, overwrite the 8 bytes in place, and restore the page to `RX`.

---

## Architecture Overview

```
main (parent)
│
├── fork → sched (scheduler, lives forever)
│             └── monitors sh->loss_history
│                 signals sh->apply_patch when plateau detected
│
└── training loop (5000 iterations)
      │
      ├── if sh->apply_patch: patch() → rewrites alpha in .text
      │
      ├── fork × (K+1) → worker 0 … worker K (one per weight)
      │                    each writes sh->grads[k]
      │                    worker 0 also writes sh->loss
      │
      ├── waitpid × (K+1)
      │
      ├── get_alpha()  ← reads the (possibly just-patched) value from .text
      │
      └── update weights:  w[k] -= alpha * grads[k]
```

All inter-process communication goes through a single `mmap(MAP_SHARED|MAP_ANONYMOUS)` region of type `Shared`.

---

## Prerequisites & Build

**Platform:** Linux x86-64 (uses `/proc/self/maps`, `mprotect`, GCC inline asm, `__builtin___clear_cache`).

**Dependencies:** standard C library + `libm`.

```bash
# Normal build
gcc -O2 -o train train_clean.c -lm

# Debug build (enables DBG_PRINTF trace output)
gcc -O2 -DDEBUG -o train train_clean.c -lm
```

> **Important:** Do not use `-O3` with link-time optimisation. The patcher depends on `get_alpha` not being inlined. The `__attribute__((noinline))` attribute handles this for the function itself, but aggressive whole-program optimisation could still interfere. `-O2` is the tested level.

Then run:

```bash
./train
```

The program expects `data.bin` in the working directory (see [Data Format](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#data-format)).

---

## Data Format

`data.bin` is a flat binary file containing exactly:

- **N × 8 bytes** — the x-values as `double` (little-endian IEEE 754), in the range `[-π, π]`
- **N × 8 bytes** — the corresponding y-values as `double`, representing `sin(x) + noise`

Total file size: `2 × N × sizeof(double)` = `2 × 100 × 8` = **1 600 bytes**.

The loader divides every x-value by π, feature-scaling the input to `[-1, 1]` before training.

A generator script might look like:

```python
import numpy as np, struct
rng = np.random.default_rng(0)
xs  = rng.uniform(-np.pi, np.pi, 100)
ys  = np.sin(xs) + rng.normal(0, 0.05, 100)
with open("data.bin","wb") as f:
    f.write(struct.pack("100d", *xs))
    f.write(struct.pack("100d", *ys))
```

For the convenience of testing, separate source file, data.c has been included. Compiling and running the compiled ELF executable automatically generates data.bin with random Gaussian noise inserted using a Box-Muller transformation, a mathematical theorem that the author first tested in [htype](github.com/lord-of-the-strings/htype). 

---

## Hyperparameters

|Macro|Value|Meaning|
|---|---|---|
|`K`|8|Polynomial degree (fits w₀ + w₁x + … + w₈x⁸)|
|`N`|100|Number of training samples|
|`EPOCH`|5 000|Number of gradient-descent iterations|
|`ALPHA`|1e-6|Minimum (floor) learning rate|
|`NOISE`|0.05|Gaussian noise std-dev on the sine (data-gen only)|
|`SCHED_INTERVAL`|50|How often (in epochs) the scheduler is signalled|
|`SCHED_HISTORY`|8|Rolling window size for plateau detection|
|`SENTINEL_A`|`0xDEADBEEFCAFEBABE`|Magic prefix in `.text` before alpha|
|`SENTINEL_B`|`0xFEEDFACEDEADC0DE`|Magic suffix in `.text` after alpha|
Since several engineering decisions were made and changed throughout the workflow, some macros are likely to be redundant. The author will be grateful to anyone who opens a pull request removing any redundancy in his source.

---

## Section-by-Section Breakdown

### Section 1 — Headers, Macros & Shared Memory

The `Shared` struct is the nervous system of the whole program. Every inter-process signal passes through it:

```c
typedef struct {
    double weights[K+1];      // model parameters, updated by main
    double grads[K+1];        // partial derivatives, written by workers
    double loss;              // MSE, written by worker 0
    int    iter;              // current epoch, written by main
    int    scheduler_flag;    // main → sched: "wake up, new loss available"
    int    patching;          // patcher → main: "hold on, I'm mid-write"
    double loss_history[8];   // rolling window maintained by sched
    int    history_idx;       // ring-buffer index
    double scheduled_alpha;   // sched → main: "use this new alpha"
    int    apply_patch;       // sched → main: "please patch now"
} Shared;
```

There are no mutexes. Synchronisation is achieved by:

- `waitpid` ensuring workers are done before main reads gradients.
- Busy-polling on `sh->patching` before reading alpha.
- `usleep` spin-waits in the scheduler on `sh->apply_patch`.

### Section 2 — Data Loader

`load()` reads `data.bin` into two file-scope static arrays `xs[N]` and `ys[N]`. They are `static` so that forked children inherit them in their address space without any extra IPC — a clean (if implicit) use of `fork`'s copy-on-read semantics.

Feature scaling (÷ π) is applied in-place after loading.

### Section 3 — The Runtime Code Patcher

This is the star of the show. See [The Patching Trick in Detail](https://claude.ai/chat/f5738b45-9406-4cf9-babd-02627a3f483a#the-patching-trick-in-detail) for the full walkthrough.

In brief: `get_alpha()` does not read alpha from a variable. It executes an `x86-64` `movabsq` instruction whose 64-bit immediate operand _is_ the learning rate, encoded as a raw IEEE 754 double baked into the instruction stream. The `patch()` function finds that immediate by scanning `.text` for the two sentinel values flanking it, then overwrites the 8 bytes in place.

### Section 4 — Worker Children

`worker(sh, k)` is the body of each short-lived child. For weight index `k` it:

1. Iterates over all N training points.
2. Evaluates the current polynomial `ŷ = Σ w[j] * x^j` using Horner-style accumulation.
3. Computes the residual `r = ŷ - y`.
4. Accumulates `grad += r * x^k` (the partial derivative of MSE w.r.t. w\[k]).
5. Writes `sh->grads[k] = 2/N * grad`.
6. If `k == 0`, also writes `sh->loss = Σ r² / N`.
7. Calls `_exit(0)` — not `exit()`, to avoid flushing stdio buffers inherited from the parent.

Workers never touch `sh->weights`; they only read it.

### Section 5 — Scheduler Child

`sched(sh)` runs in an infinite loop:

1. Spin-sleeps on `sh->scheduler_flag == 0`.
2. When woken, records `sh->loss` into the rolling `loss_history` ring buffer.
3. Once the buffer is full (after `SCHED_INTERVAL × SCHED_HISTORY` epochs), computes the improvement ratio across the window: `(max_loss - min_loss) / max_loss`.
4. If improvement falls below **2%**, the learning rate is halved (floored at `ALPHA`), and the scheduler sets `sh->scheduled_alpha` and `sh->apply_patch = 1`, then spin-waits for main to complete the patch and clear `apply_patch`. The comments mention 1%, but 2% has been found to be optimal to wake up the scheduler to call the patcher mid-execution for eye candy. 1% is good for a real application so the comment stays as is, but during testing it showed that it never needed to be patched at 1%.

### Section 6 — `main`

`main` has three logical parts:

**Startup:** load data, `mmap` the shared region, initialise weights with a fixed seed (`srand(99)`), zero out gradients and loss history, fork the scheduler.

**Training loop:** for each of the 5 000 epochs —

- Apply any pending patch (delegated from scheduler).
- Fork K+1 worker children.
- `waitpid` all of them.
- Spin until `sh->patching == 0` (guard against a patch mid-epoch).
- Read `alpha` via `get_alpha()`, preceded by a compiler memory barrier (`asm volatile("" ::: "memory")`) to prevent the compiler from hoisting the call out of the loop.
- Update weights: `w[k] -= alpha * grads[k]`.
- Every `SCHED_INTERVAL` iterations: set `scheduler_flag = 1` and print a progress line.

**Cleanup:** `SIGTERM` the scheduler, `waitpid` it, print final weights and loss, `munmap`.

---

## Process Topology

```
Epoch N:
  main forks:
    worker[0]  → computes ∂L/∂w₀, sh->loss
    worker[1]  → computes ∂L/∂w₁
    ...
    worker[8]  → computes ∂L/∂w₈
  main waitpid(all workers)
  main reads alpha, updates weights
  (every 50 epochs) main signals sched via sh->scheduler_flag=1

sched (running concurrently):
  wakes on scheduler_flag
  updates loss_history ring buffer
  if plateau: sets sh->apply_patch=1, waits
  main patches .text, clears sh->apply_patch
  sched resumes
```

Total concurrent processes at peak: `1 (main) + 1 (sched) + 9 (workers)` = **11 processes**.

---

## The Patching Trick in Detail

### 1. Embedding alpha in `.text`

`get_alpha()` contains this inline assembly:

```asm
jmp   1f
.quad 0xDEADBEEFCAFEBABE    ← SENTINEL_A (8 bytes, skipped by jmp)
1:
movabsq $0x3F1A36E2EB1C432D, %rax   ← alpha as raw IEEE 754 bits
movq    %rax, <output operand>
jmp   2f
.quad 0xFEEDFACEDEADC0DE    ← SENTINEL_B (8 bytes, skipped by jmp)
2:
```

`0x3F1A36E2EB1C432D` is the hex encoding of the initial alpha value (~1e-4). The two `.quad` directives inject 8-byte magic constants at known offsets relative to the `movabsq`'s immediate field.

The `jmp` instructions ensure the sentinels are never executed as code.

### 2. Finding the immediate at runtime

`get_bounds()` parses `/proc/self/maps` to find the executable mapping (`r-xp` or `rwxp`) that contains `get_alpha`. `patch()` then scans that region byte by byte, looking for SENTINEL_A at offset +2 from the current pointer (because the `jmp 1f` opcode is 2 bytes). Once found, it checks that SENTINEL_B appears at a fixed offset (+27 bytes from the scan position). When both are located, the `movabsq` immediate sits at `SENTINEL_A_addr + 8 + 2` bytes.

### 3. Writing the new value

```c
mprotect(page, pagesize, PROT_READ|PROT_WRITE|PROT_EXEC);
memcpy(imm_addr, &new_alpha, sizeof(double));
__builtin___clear_cache(imm_addr, imm_addr + sizeof(double));
mprotect(page, pagesize, PROT_READ|PROT_EXEC);
```

`__builtin___clear_cache` is a GCC built-in that emits the necessary instructions to flush the I-cache / D-cache coherency, ensuring the CPU doesn't execute stale cached instructions after the write.

### 4. Why the compiler memory barrier matters

Before reading alpha back in the training loop:

```c
__asm__ volatile("" ::: "memory");
double alpha = get_alpha();
```

The barrier prevents the compiler from treating `get_alpha()` as a pure function and moving the call outside the loop (LICM — Loop-Invariant Code Motion). Without it, a smart compiler could evaluate alpha once before the loop starts, defeating the entire patching mechanism.

---

## Scheduler Logic

The scheduler uses a **sliding window plateau detector**:

- Window size: 8 loss samples, recorded every 50 epochs → covers the last 400 epochs.
- Improvement metric: `(max − min) / max` across the window.
- Threshold: if improvement < 2%, the learning rate is halved.
- Floor: `ALPHA = 1e-6` — the rate is never lowered below this.

This is a simple but effective heuristic: if loss hasn't improved by at least 2% over the last ~400 epochs, the gradient steps are too large and the optimiser is either oscillating or stuck on a flat region.

The scheduler does **not** call `patch()` itself — it only sets flags and waits. The actual patching is delegated to the parent (`main`) to avoid two processes competing for the same `.text` page.

---

## Known Quirks & Limitations

**Linux x86-64 only.** The inline assembly, `/proc/self/maps` parsing, and I-cache flush are all platform-specific.

**No mutex around shared memory.** This works because:

- Workers only write their own `grads[k]` slot (no aliasing).
- The scheduler only reads `loss` and writes non-overlapping fields.
- `waitpid` acts as the synchronisation barrier for the hot path.

Removing or reordering those implicit barriers would introduce data races.

**WEXITSTATUS not checked.** `waitpid(pids[k], NULL, 0)` discards the worker's exit status. A worker crash would silently leave stale gradient values in shared memory.

**The patcher will `exit(1)` if the sentinel pattern is not found.** This can happen if the binary is compiled with an optimisation level that causes `get_alpha` to be reorganised unexpectedly (e.g., certain PGO or LTO passes).

**Scheduler wakeup is coarse.** The scheduler is signalled once every `SCHED_INTERVAL = 50` epochs, not every epoch. Loss history therefore has 50-epoch resolution.

---

## Performance & Efficiency

### vs. Vanilla Python

The C implementation will comfortably outperform a naïve Python equivalent — tight loops over 100 points, 9 weights, 5 000 epochs, with no interpreter overhead, no boxing of floats, and no GIL contention. For a pure Python gradient descent loop doing the same arithmetic, expect a 50–200× wall-clock gap.

### vs. Vectorised NumPy / C

This is the more honest comparison. A vectorised implementation — whether NumPy in Python or explicit SIMD intrinsics in C — would compute the full gradient in a single matrix multiply and update all K+1 weights in one vector operation per epoch. That maps directly onto AVX2/AVX-512 units and keeps the CPU's execution ports saturated. The multiprocess design here does the opposite: it parallelises across weights (9 scalar workers) rather than across data points, so it never gets to use SIMD at all. For N=100 and K=8, the cost of 9 `fork`/`_exit` pairs — each involving a page-table clone, scheduler round-trip, and process teardown — almost certainly _exceeds_ the cost of the arithmetic being parallelised. A single-threaded vectorised loop would likely beat this program on a modern CPU despite being "less parallel" on paper, although that has not been formally tested yet. The author will be grateful to any kind stranger who dedicates his time to doing that.

### The `.text` Embedding and the I-Cache Problem

Storing alpha as a raw IEEE 754 immediate inside the instruction stream is the program's sharpest self-inflicted performance wound. Modern CPUs maintain separate L1 caches for instructions (I-cache) and data (D-cache). Under normal operation the I-cache holds hot code paths in a decoded, pre-fetched form — the CPU reads ahead, predicts branches, and queues up decoded micro-ops so that execution units are never starved. When `patch()` calls `mprotect(..., PROT_READ|PROT_WRITE|PROT_EXEC)` and writes 8 bytes into the live instruction stream, the CPU detects a **self-modifying code** event. It must:

1. Flush the pipeline for the affected region.
2. Invalidate the I-cache lines covering the patched page.
3. Force a cold re-fetch of those instructions from L1 D-cache or memory.

builtin___clear_cache makes this explicit, but the cost is unavoidable regardless. For a value that changes at most a handful of times across 5 000 epochs, the _amortised_ overhead per patch event is negligible. But the mechanism itself disables one of the CPU's most important performance features — instruction-level pre-fetching — for the duration of the patch, and reloads cache lines that would otherwise stay warm across thousands of tight inner-loop iterations. A global `double alpha` updated by the scheduler via the existing shared-memory region would have zero pipeline disruption, cost one `movsd` from L1, and be JIT-visible to the compiler for potential constant-propagation. The `.text` approach is a clever trick; the author did this just because he could.

So why then did the author do this? Anyone who has read AlephOne's legendary article, *Smashing the Stack for Fun and for Profit*, the literal Bible for ethical hackers, would have encountered a line on the first or second page saying that .text is marked read-only by default. The author wanted to force his will on the defaults and that's why this was done. As for the child processes gimmick, it was to test performance rise by parallelism.

-------------------------------

## Debugging

Build with `-DDEBUG` to enable `DBG_PRINTF` traces:

```bash
gcc -O2 -DDEBUG -o train train_clean.c -lm
```

This prints:

- The `.text` scan range and `get_alpha`'s address.
- The offset at which SENTINEL_A is found, plus a 32-byte hex dump of the surrounding bytes.
- Worker fork/start events.
- Each call to `get_alpha` and the value returned.

There are also two commented-out lines in `main` for a quick sanity-check of the patcher in isolation:

```c
// patch(sh, 0.01);
// printf("Patcher OK\n");
```

Uncomment these to verify the patcher finds its target before starting the training loop.

---

## License

MIT — as declared by the author in the final line of the source file.

> _"Proudly hand-typed in vim by Aadity Setu (@lord-of-the-strings), all rights reserved as per MIT licence."_
