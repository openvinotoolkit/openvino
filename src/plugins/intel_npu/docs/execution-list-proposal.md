# [OV][NPU] Context-scoped execution lists for chained inference requests

## Context

When NPUW runs an LLM such as Qwen3-4B with `NPUW_UNFOLD_IREQS=YES`, every decoder block gets its own
NPU infer request. All of them come from the same compiled model, and each one has its own weights
passed as inputs. The per-block weights are bound once, in the `UnfoldInferRequest` constructor
(`unpack_closure`, `npuw/unfold_sync_infer_request.cpp:82-93`), so a request's arguments are mostly
fixed between inferences.

With `NPUW_FUNCALL_ASYNC` (on by default for LLMs, `npuw/llm_compiled_model.cpp:258`), NPUW calls
`start_async()` on every block of a function group back to back, then waits on all of them
(`npuw/unfold_sync_infer_request.cpp:117-141`). To keep the blocks in order, NPUW forces
`NPU_RUN_INFERENCES_SEQUENTIALLY=YES` on NPU submodels (`npuw/compiled_model.cpp:231`, `:2452`).

In the NPU plugin that flag works as follows:

- On command-queue extension 1.1 or newer, it becomes `ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC` on the
  queue (`compiler_adapter/src/graph.cpp:196-205`). Older drivers chain requests with events instead
  (`backend/src/zero_pipeline.cpp:262-290`).
- The compiled model switches to single-thread start and wait executors
  (`plugin/src/compiled_model.cpp:236-253`).

Every `start_async()` on a block therefore costs:

1. a thread hop to the start executor;
2. `prepare_inputs()`, which takes the graph mutex and checks every input (including all the weight
   inputs) for memory changes;
3. one `zeCommandQueueExecuteCommandLists(queue, 1, &list, fence)` call, i.e. one driver job
   (`utils/src/zero/zero_wrappers.cpp:350-357`);
4. a thread hop to the wait executor and one `Fence::hostSynchronize()`.

For 36 decoder blocks that adds up to 36 submissions, 36 fences and 72 thread hops per generated
token. Host-side costs mostly overlap with device execution, because NPUW submits ahead of the
device. The device-side cost is the gap between jobs (firmware scheduling and per-job setup), paid
once per block. Recording all blocks into one command list removes that gap.

### Why the current plugin cannot do this

- `CommandList` holds a single mutable command id (`_command_id`,
  `utils/include/intel_npu/utils/zero/zero_wrappers.hpp:114`), taken once in the constructor. Only
  one mutable command per list can have its arguments updated.
- Nothing in the plugin appends more than one `appendGraphExecute` to a list. Plugin batching also
  uses N separate command lists.
- The plugin records command lists lazily, on the first inference: `prepare_inputs()` calls
  `setup_pipeline()` (`backend/src/zero_infer_request.cpp:1004-1010`), not request creation. At
  that point NPUW has already bound the weights and the links between blocks.
- OpenVINO has no API that groups infer requests. `ICompiledModel::create_infer_request()` takes no
  arguments, and infer requests have no property API.

## Proposal

Add an **execution list** to the dev API. A plugin's remote context creates it, requests created
from compiled models bound to that context can be appended to it, and the whole list is submitted as
one unit.

The remote context is the right scope:

- A compiled model is already tied to a context.
- NPUW compiles its submodels with `core->compile_model(model, "NPU", cfg)`, so they all share the
  default NPU context.
- `ICompiledModel::get_context()` already exists in the dev API
  (`inference/dev_api/openvino/runtime/icompiled_model.hpp:135`).

### API sketch (dev API)

```cpp
// inference/dev_api/openvino/runtime/iremote_context.hpp
class OPENVINO_RUNTIME_API IRemoteContext : public std::enable_shared_from_this<IRemoteContext> {
public:
    // ...
    /// Default implementation returns a core-provided SequentialExecutionList.
    virtual std::shared_ptr<ov::IExecutionList> create_execution_list(const ov::AnyMap& params = {});
};

// inference/dev_api/openvino/runtime/iexecution_list.hpp
class OPENVINO_RUNTIME_API IExecutionList {
public:
    virtual ~IExecutionList() = default;

    /// Appends a request. Execution order follows append order.
    virtual void append(const ov::SoPtr<ov::IAsyncInferRequest>& request) = 0;

    virtual void start_async() = 0;
    virtual void wait() = 0;
    virtual bool wait_for(const std::chrono::milliseconds& timeout) = 0;
    virtual void set_callback(std::function<void(std::exception_ptr)> callback) = 0;
};
```

NPUW usage:

```cpp
auto context = subrequests.front()->get_compiled_model()->get_context();
auto list = context->create_execution_list();
for (auto&& request : function_group_requests) {
    list->append(request);
}
// per inference
list->start_async();
list->wait();
```

### Design decisions

**No separate command queue object.** In the NPU plugin the queue is not a free-standing object. It
comes from compiled-model config: priority, turbo, workload type and device-sync all go into
`CommandQueueDesc` (`compiler_adapter/src/graph.cpp:209-215`), and the queue is then taken from
`ZeroCmdQueuePool`. A user-visible queue would conflict with those per-model properties. The rule
is that the list runs on its members' queue: if members disagree, `append` fails, or the list takes
its own priority and workload-type properties through `params`.

GPU already exposes queue sharing for interop, but as a context parameter (`ov::intel_gpu::ocl_queue`,
`inference/include/openvino/runtime/intel_gpu/remote_properties.hpp:100`), not as an object. A queue
object only becomes worth adding if an application needs to share it with its own Level Zero queue.

**Named after the OpenVINO meaning, not Level Zero.** "These requests, in this order, submitted as one
unit." A name like `RemoteCommandList` suggests it only works on L0-style devices.

**Core provides a default implementation.** `SequentialExecutionList` calls `start_async()` and `wait()`
on each member in order, which is exactly what NPUW does today. So:

- CPU and GPU work unchanged, and NPUW has a single code path with no device checks.
- GPU can add a native version later, for example by enqueuing all members on its in-order OpenCL
  queue without host syncs in between, or with `cl_khr_command_buffer`.
- The NPU plugin overrides it with a single-command-list version.

**Dev API first, public API later.** A public `ov::` class means C and Python bindings, docs, API and
ABI stability, and review by every plugin owner. NPUW is the only consumer today and it already uses
the dev API. Dev-API plugins are always built with the same OpenVINO version, so adding a virtual to
`IRemoteContext` has a much lower bar. Making it public can be proposed once the speed-up is proven.

## Semantics

1. **Live bindings.** Level Zero captures arguments when a command is appended, but OpenVINO lets
   callers use `set_tensor` at any time. The list holds references to its requests. On each
   `start_async`, the implementation syncs every member's current tensors. The NPU version updates
   only the members whose tensors changed, using their command ids. NPUW needs this because it
   rebinds KV-cache and global inputs on every inference.
2. **Member state.** While the list is running, its members are busy. Their own `start_async`,
   `set_tensor` and similar calls throw `REQUEST_BUSY`. Outputs are read with each member's
   `get_tensor` after `wait()`. A request may belong to several lists, but only one can run at a
   time.
3. **Host-side work.** A list contains device work only. Host work that the NPU plugin does in
   `infer_async` is handled as follows:
   - input copies and shape-tensor writes happen before submission;
   - output copies to user buffers that are not Level Zero memory happen after completion, inside
     `wait()`;
   - dynamic pipelines and plugin batching (N command lists) are rejected by `append`.

   NPUW splits its lists wherever it does host work, such as `needs_copy` in `bind_global_params`,
   spatial execution or host flash attention.
4. **Membership rules.** Only requests from the same context can join. For NPU they must also have
   the same queue settings. Members may come from different compiled models, because
   `appendGraphExecute` accepts any graph handle in the same Level Zero context. That allows the
   embedding, all decoder blocks and the tail to become one submission.
5. **Profiling and errors.** Each member keeps its own profiling query, so per-member
   `get_profiling_info` keeps working. Errors are reported per list through `wait()` or the callback.
   A failure cannot be attributed to a specific member.
6. **Lifetime.** The list holds `shared_ptr`s to its members, so a member cannot be destroyed while it
   is in a list.

## NPU plugin implementation

- `RemoteContextImpl::create_execution_list()` returns a `ZeroExecutionList`.
- `ZeroInferRequest::infer_async()` is split so that `prepare_inputs()` and `prepare_outputs()` can
  run without `Pipeline::push()`. `ZeroExecutionList` calls them for each member.
- Recording happens on the first `start_async()` and builds one `CommandList`. For each member k:
  1. set the member's graph arguments from its Level Zero tensors (under the graph mutex, as
     `prepare_inputs()` already does);
  2. `zeCommandListGetNextCommandIdExp` → `id_k`;
  3. `appendGraphExecute` with the member's graph handle and profiling query;
  4. `appendBarrier`, because the L0 spec does not guarantee ordering in a regular list without
     `ZE_COMMAND_LIST_FLAG_IN_ORDER`.

  A single fence goes at the end. Each inference is then one `zeCommandQueueExecuteCommandLists`
  call and one fence wait.
- On later inferences, changed member tensors are applied with `zeCommandListUpdateMutableCommandsExp`
  using `id_k`. Structural changes (a member's graph or argument count) trigger re-recording.
- `CommandList` stores one command id per mutable command instead of the single `_command_id`.
  `updateMutableCommandList` and `updateMutableCommandListWithStrides` take the command id.
- Members of a list do not need `RUN_INFERENCES_SEQUENTIALLY`, because ordering comes from the
  barriers inside the list. NPUW still waits between lists, so lists stay ordered relative to each
  other.

### NPUW changes

- `UnfoldInferRequest` builds one execution list per run of consecutive blocks from the same
  function. The existing `do_async` loop already finds these runs by `replaced_by`. Later, the
  head, the blocks and the tail can share one list.
- `bind_global_params` and `bind_global_results` run for all members before `list->start_async()`.
  NPUW loses the current overlap where block i+1 is prepared on the host while block i runs on the
  device. For unfold this is mostly KV-cache and global inputs, which should be cheap.
- With the core default implementation, non-NPU devices need no special handling.

## Alternatives considered

**NPUW uses Level Zero directly.** NPUW would need graph handles, driver argument indices, Level Zero
tensors and mutable command ids, all of which live inside `ZeroInferRequest` and `Pipeline`. It
would duplicate the backend and tie NPUW to NPU, which makes fallback to other devices harder.
Rejected.

**One infer request representing N executions.** An internal compile property such as
`NPU_INTERNAL_CHAINED_EXECUTION=YES` makes `CompiledModel::create_infer_request()` return a
`ZeroChainedInferRequest`. `ZeroDynamicInferRequest` is the precedent for choosing a request type
inside the plugin. Tensors for each execution are passed with `set_tensors(port, {t0 … tN-1})`, and
the plugin's per-execution outputs are returned by `get_tensors(output)`.
`IAsyncInferRequest::set_tensors` forwards to the plugin without validation
(`inference/src/dev/iasync_infer_request.cpp:238`). This needs no core change. It gives
`set_tensors` a second meaning ("per execution" rather than "per batch item"), it only covers one
compiled model per chain, and it only works on NPU. This is a good fallback if a core change is not
acceptable.

**N requests sharing a list through a compiled-model property.** A chain handle (`ov::Any` holding
a `shared_ptr`) is set on the compiled model before each `create_infer_request()`. This is stateful
and races with other NPUW requests that use the same compiled model. It is also unclear which
member submits the shared list. Rejected.

**Plugin batching (`set_tensors` with a batch dimension).** Batch items are independent and need a
batch dimension and a layout with N. Chained blocks depend on each other. Rejected.

**Level Zero graph capture (`zeCommandListBeginGraphCaptureExp` / `zeGraphInstantiateExt`).** Needs
immediate command lists and targets GPU. Assume it is not available on NPU unless the driver team
confirms otherwise.

**Compiling the repeated block as a loop.** Removes the problem at its source, but it is a much
larger compiler-side change. Long term only.

## Open questions

1. **Driver:** does the NPU driver support several mutable graph commands in one command list? Is
   there a limit on graph executes per list or per job?
2. **Driver:** does submitting several command lists in one `zeCommandQueueExecuteCommandLists` call
   produce one firmware job or several? If one, a cheaper intermediate step is to keep one list per
   member and submit them together.
3. **Preemption:** one large job may only be preemptible at job boundaries. That could hurt other
   models sharing the NPU, especially with `SHARED_COMMON_QUEUE` or model priorities.
4. **Core API owners:** is a new virtual on `IRemoteContext`, with a default implementation,
   acceptable in the dev API?

## Proof of concept

An NPU-only version is implemented for measurement, separate from the standard inference flow:

- `src/common/include/intel_npu/common/execution_list.hpp`: `IExecutionList` (`append`, `start_async`,
  `wait`) and `IExecutionListFactory`. They use only virtual methods, so they can be used from outside
  the plugin library.
- `RemoteContextImpl` implements `IExecutionListFactory`. Its list maps each `ov::IAsyncInferRequest`
  to the backend `ZeroInferRequest` through `AsyncInferRequest::get_infer_request()`.
- `backend/src/zero_execution_list.cpp`: `ZeroExecutionList` records all requests into one command list,
  with a barrier between executions, and submits it with one fence. Before each submission it compares
  every request's tensor addresses with the recorded ones and updates only what changed, by command id.
- `CommandList` gained `getNextCommandId()` and an `updateMutableCommandList` overload that takes a
  command id.
- `ZeroInferRequest::get_result()` was split so that the output handling (`update_user_outputs()`) can
  run without waiting on the request's own fence.
- `tools/execution-list-bench` has an `execution-list` mode that uses the list. The public API doesn't
  expose the implementation objects, so the app reaches them through access shims.

Not handled by the proof of concept: dynamic models, plugin batching, strided tensors, profiling, and
preventing a request from being used on its own while its list is running.

## Next steps

1. **Measure.** Run Qwen3-4B generate with `PERF_COUNT=YES` and `NPU_PROFILING_TYPE=INFER`, or with
   ITT/VTune using the existing `Pipeline::push` and `Pipeline::pull` markers. Compare per-block
   device time with the gaps between blocks.
2. **Ask the driver team** open questions 1 and 2.
3. **Prototype NPU-only** behind a debug flag. NPUW casts the context to the NPU plugin's
   `RemoteContextImpl` (NPUW already casts to `::intel_npu::ZeroRemoteTensor` in
   `npuw/whisper/whisper_infer_request.cpp:44`). Implement `ZeroExecutionList` and the multi-id
   `CommandList`, then compare tokens per second.
4. **If the gain holds**, add `create_execution_list()` and `SequentialExecutionList` to the core dev
   API, and remove the cast from NPUW.
