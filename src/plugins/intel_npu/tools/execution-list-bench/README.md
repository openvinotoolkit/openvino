# execution-list-bench

Measures the cost of running a chain of infer requests of the same compiled model, the way NPUW runs
repeated decoder blocks with `NPUW_UNFOLD_IREQS=YES`. It compares that with a single model that has all
blocks merged, which shows how the device runs the same work without per-request overhead.

Background and the proposed execution list are described in
[execution-list-proposal.md](../../docs/execution-list-proposal.md).

## What it runs

The app generates the models with the OpenVINO API, so no model files are needed.

- **Block model**: `x[1, seq, hidden] -> MatMul(w_up) -> Swish -> MatMul(w_down) -> Add(x)`. The weights
  `w_up[ffn, hidden]` and `w_down[hidden, ffn]` are inputs, like the closures NPUW passes to a repeated
  block. The default sizes match the Qwen3-4B MLP.
- **Merged model**: `-blocks` copies of the block chained into one model. By default every block's weights
  are inputs, so it does exactly the same work as the chain of requests. With `-ref_weights const` they
  are constants instead, and the compiler sees them.

All modes use the same input and the same weight tensors. When the device's default context supports
it, the tensors are allocated with `create_host_tensor`, so the plugin uses them without copies.

## Modes

| Mode | Behaviour | Submissions per iteration |
| --- | --- | --- |
| `unfold-async` | One request per block. The weights are bound once, and each request reads the previous request's output tensor. All requests are started, then all are waited on (`NPUW_FUNCALL_ASYNC=YES`). | `blocks` |
| `unfold-sync` | The same requests, but each one is started and waited on before the next. | `blocks` |
| `execution-list` | The same requests, appended to one execution list created by the NPU remote context. All blocks are recorded into one command list and submitted with one fence. NPU only. | 1 |
| `reference` | The merged model, one request. | 1 |

For the unfold modes on NPU, the block model is compiled with `NPU_RUN_INFERENCES_SEQUENTIALLY=YES`, as
NPUW does. Other devices don't guarantee that requests run in order, so `unfold-async` results may be
wrong there. The `execution-list` mode compiles its own copy of the block model without that property,
because the list orders the requests itself.

`-mode all` (the default) runs every mode; `execution-list` is included only on NPU.

## Execution list proof of concept

The `execution-list` mode uses an interface internal to the NPU plugin,
`src/plugins/intel_npu/src/common/include/intel_npu/common/execution_list.hpp`:

- `RemoteContextImpl` implements `IExecutionListFactory::create_execution_list()`.
- `IExecutionList::append()` takes the plugin's `ov::IAsyncInferRequest`; `start_async()` and `wait()`
  run the whole list.
- The backend (`ZeroExecutionList`) records, for each request: its graph arguments, a mutable command id,
  the graph execute and a barrier. If a request's tensors move later, only that request's command is
  updated.

The public API doesn't expose the implementation objects behind `ov::RemoteContext` and
`ov::InferRequest`, so the app reaches them through access shims in `main.cpp`. This is for the proof
of concept only.

Limitations: static models only, no plugin batching, no strided tensors and no profiling. While a list
is running, its requests must not be used on their own.

## Output

For each mode the app prints the median, mean, min, p90 and max time of one iteration (the whole chain).
It also prints:

- **overhead**: the median minus the reference median;
- **overhead/blk**: the overhead divided by the number of blocks;
- **rel diff**: the max-abs difference between the mode's output and the reference output, divided by
  the reference's max-abs value.

## Accuracy check

With `-check` (on by default), the app computes a reference output once, before any mode runs:

- It runs the merged model a single time on `-ref_device` (CPU by default) and keeps the output.
- `-config` properties are only passed to that run if `-ref_device` is the device under test.
- An empty `-ref_device ""` computes the reference on the device under test.

The individual inferences are not checked. Each mode is checked once, after its warm-up and measured
iterations have all run: the output left by the last iteration is compared with the fixed reference.
Every iteration computes the same output from the same input, so this catches iterations that corrupt
the result, for example stale tensor addresses in a reused command list. The `reference` mode is checked
the same way.

The app exits with an error if `rel diff` for any mode is larger than `-tolerance`.

## Rebinding tensors

By default a chain's tensors never move, so an execution list is recorded once and then replayed
unchanged. With `-rebind`, before every iteration the block requests switch between two sets of
output tensors, and each block's input follows the previous block's output. This mirrors NPUW
rebinding the KV cache between tokens. The list then has to update the commands of all blocks
through their mutable command ids. The result doesn't change, so the accuracy check after the
last iteration shows whether those updates were applied.

## Examples

```sh
# Qwen3-4B-sized MLP blocks, all modes
execution-list-bench -d NPU

# Smaller blocks, where per-submission overhead dominates
execution-list-bench -d NPU -hidden 1024 -ffn 4096

# Prefill-like input, reference with weights as constants
execution-list-bench -d NPU -seq 128 -ref_weights const

# Extra properties for every compiled model
execution-list-bench -d NPU -config "NPU_TURBO=YES"
```

Run `execution-list-bench -h` for all options.
