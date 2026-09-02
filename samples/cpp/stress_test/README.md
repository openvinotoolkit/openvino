# stress_test

A native C++ stress test for the OpenVINO Runtime targeting stability issues
seen under heavy concurrent usage: Level Zero errors, memory access
violations, multi-thread races, and driver hangs caused by rapid load/unload
of the Level Zero NPU driver.

Works with LLM models (`input_ids`/`attention_mask`), non-LLM models (BERT,
audio encoder, image encoder), and classic vision models (classification,
detection). All inference uses synchronous `infer_req.infer()` with named
tensor access (`set_tensor` / `get_tensor`), matching the pattern used by
the aicore backend.

---

## Build

```bash
cmake -B build -S samples/cpp \
      -DOpenVINO_DIR=<openvino_install>/runtime/cmake
cmake --build build --config Release --target stress_test
```

The directory is also picked up automatically when building through the
top-level samples CMake.

---

## Usage

```
stress_test -m <model> [options]
```

### Required

| Option | Description |
|--------|-------------|
| `-m <path>` | Model file (`.xml`, `.onnx`, or any format accepted by `core.read_model()`). |

### Optional

| Option | Default | Description |
|--------|---------|-------------|
| `-d <device>` | `CPU` | Target device: `CPU`, `NPU`, `GPU`, `AUTO`, etc. |
| `-t <N>` | `4` | Worker threads per scenario. |
| `-n <N>` | `50` | Iterations per scenario per thread. Ignored when `--duration` is set. |
| `--duration <s>` | — | Run each scenario for `<s>` seconds instead of a fixed iteration count. |
| `--only <name>` | — | Run a single scenario by name (see table below). When omitted all scenarios run in sequence. |
| `--log-dir <dir>` | `stress_logs` | Directory for firmware and kernel log snapshots. Created automatically if absent. |
| `--hang-timeout <s>` | `60` | Watchdog timeout: if no `infer()` returns for this many seconds, saves logs and calls `_Exit(1)`. Targets the "NPU driver stuck" failure mode. |
| `--fw-log <path>` | `/sys/kernel/debug/accel/0000:00:0b.0/fw_log` | Sysfs path to the NPU firmware log. Adjust the PCI address for the target system. |

---

## Scenarios

| Name | `--only` value | Driver operations exercised |
|------|---------------|----------------------------|
| Load / Unload | `load_unload` | Concurrent context create + infer + context delete |
| Parallel Inference | `parallel` | N concurrent command-queue submissions on one context |
| Concurrent Load + Infer | `concurrent` | Context create/delete racing active queue submission |
| Import / Export | `import_export` | Parallel blob import → context create → infer → delete |
| New Infer Mid-Flight | `mid_flight` | Concurrent queue submission + `cancel()` on live request |
| Memory Stress | `memory` | N simultaneous context creates, concurrent infer, N simultaneous deletes |
| Destroy Live Request | `destroy_live_req` | Context handle deleted while command queue is in flight |
| Multiple Core Instances | `multi_core` | N threads each open + close their own Level Zero connection |

### `load_unload`
`-t` threads each repeatedly compile the model, create an `InferRequest`,
run one inference, then destroy everything. Stresses driver load/unload and
resource tracking under concurrent compilation.

### `parallel`
`-t` threads share one `CompiledModel`; each thread owns its own
`InferRequest` and calls `infer()` in a tight loop. For LLM models the
sequence length varies per iteration to cover both prefill (long sequence)
and decode (single token) paths.

### `concurrent`
Two threads run simultaneously:
- **Load thread** — repeatedly compiles and destroys models.
- **Infer thread** — continuously runs inference on a pre-compiled model.

Catches races between driver init/teardown and active kernel execution, a
common source of Level Zero hangs on NPU.

### `import_export`
Compiles once, exports the blob to an in-memory buffer, then `-t` threads
each repeatedly import the blob, create an `InferRequest`, and run
inference. Skipped with a message if the device does not support
`export_model`.

### `mid_flight`
Per iteration, three threads are spawned:
- **Thread A** — sets inputs and calls `infer()`.
- **Thread B** — immediately starts its own `infer()` on a second
  `InferRequest` from the same `CompiledModel`, without waiting for A.
- **Thread C** — calls `cancel()` on Thread A's request after 2 ms.

Verifies that concurrent inference and cancellation do not corrupt shared
device state.

### `memory`
Loads `-t` `CompiledModel` instances simultaneously, runs inference on all
concurrently, then destroys all at once. Repeated for `-n` rounds. Stresses
device memory allocation, handle-table limits, and teardown ordering.

### `destroy_live_req`
Each iteration compiles a fresh model, creates an `InferRequest`, starts
`infer()` in a background thread, then immediately drops the `CompiledModel`
handle via `shared_ptr::reset()`. The `InferRequest` holds its own internal
reference so inference must complete cleanly. Exercises driver-side handle
and context lifetime tracking — a common source of Level Zero use-after-free
errors on NPU.

### `multi_core`
`-t` threads each create their own `ov::Core`, open a fresh device
connection, compile the model, run one inference, then destroy the `Core`
and all associated objects at scope exit — all threads running
simultaneously. Directly targets the system-crash scenario caused by
frequent loading and unloading of the Level Zero loader and NPU driver.

## Output

Each scenario prints a summary on completion:

```
[parallel_infer]  pass=800  fail=0  cancelled=0
  ERR: parallel[2]: <exception message>
```

- `pass` — inferences that completed successfully.
- `fail` — inferences that threw an unexpected exception.
- `cancelled` — inferences interrupted by `cancel()` in `mid_flight`;
  these are expected and do not count as failures.
- Up to 64 error messages are printed per scenario.

Press **Ctrl-C** to stop early; in-progress iterations finish before exit.

---

## Log files

For each scenario run, pairs of files are saved to `--log-dir`:

| File pattern | When captured |
|---|---|
| `baseline_{fw_log,dmesg}.txt` | Once, before any scenario runs. |
| `<scenario>_start_{fw_log,dmesg}.txt` | Immediately before each scenario. |
| `<scenario>_end_{fw_log,dmesg}.txt` | Immediately after each scenario. |
| `hang_<scenario>_{fw_log,dmesg}.txt` | By the hang watchdog, just before forced exit. |

On **Linux** and **Android**, kernel messages are captured via `dmesg(1)`.
The NPU firmware log is read from `--fw-log` (requires debugfs to be mounted;
verify the PCI address matches the target system with `ls /sys/kernel/debug/accel/`).

---

## Examples

```bash
# Run all 8 scenarios on NPU, 8 threads, 100 iterations each
stress_test -m model.xml -d NPU -t 8 -n 100

# Time-bounded full run (30 s per scenario)
stress_test -m model.xml -d NPU -t 8 --duration 30

# Classic vision model on CPU, all scenarios
stress_test -m resnet50.xml -d CPU -t 4 -n 200

# LLM prefill/decode stress, parallel inference
stress_test -m llm_decoder.xml -d NPU -t 4 --only parallel -n 500

# LLM mid-flight cancellation
stress_test -m llm_decoder.xml -d NPU -t 4 --only mid_flight -n 200

# Non-LLM (BERT / audio encoder), concurrent load + infer
stress_test -m bert.xml -d NPU --only concurrent -n 300

# Import/export round-trip under parallel load
stress_test -m model.xml -d CPU -t 8 --only import_export -n 100

# NPU hang watchdog: 30 s timeout, custom log directory
stress_test -m model.xml -d NPU -t 4 -n 100 \
    --hang-timeout 30 --log-dir /tmp/npu_stress_logs

# Override fw_log path when PCI address differs from default
stress_test -m model.xml -d NPU --only parallel -n 500 \
    --fw-log /sys/kernel/debug/accel/0000:00:0e.0/fw_log

# NPU handle tracking: destroy CompiledModel while InferRequest is live
stress_test -m model.xml -d NPU -t 1 --only destroy_live_req -n 200

# Level Zero loader crash scenario: N Core instances created/destroyed at once
stress_test -m model.xml -d NPU -t 8 -n 50 --only multi_core

# Memory pressure: N models loaded, inferred, and unloaded simultaneously
stress_test -m model.xml -d NPU -t 16 -n 20 --only memory
```
