# OpenCL Kernel Synchronization in GPU Plugin

This document describes how the GPU plugin synchronizes kernel execution within a `stream` (OpenCL command queue) and why a particular method is chosen for a given device/config.

## Overview

Kernel dependencies inside a `cldnn::network` are expressed as a graph of `event::ptr` objects. When a primitive is executed, its dependency events must complete before the primitive's own kernel starts. `cldnn::stream` (see [stream.hpp](../include/intel_gpu/runtime/stream.hpp)) supports three synchronization strategies, defined by `SyncMethods`:

```cpp
enum class SyncMethods {
    events   = 0, // build dependency graph using real cl_event wait-lists
    barriers = 1, // enqueue clEnqueueBarrierWithWaitList between dependent kernels
    none     = 2  // rely on in-order queue semantics, no explicit sync
};
```

The method is decided once per stream by `stream::get_expected_sync_method()` ([stream.cpp](../src/runtime/stream.cpp)):

```cpp
SyncMethods stream::get_expected_sync_method(const ExecutionConfig& config) {
    auto profiling = config.get_enable_profiling();
    auto queue_type = config.get_queue_type();
    return profiling ? SyncMethods::events
         : queue_type == QueueTypes::out_of_order ? SyncMethods::barriers
                                                   : SyncMethods::none;
}
```

## Decision Table

| Device type | `queue_type` | Sync method (no profiling) | Sync method (profiling enabled) |
|---|---|---|---|
| Systolic array GPU (`supports_immad == true`) | in-order | `none` | `events` |
| Non-systolic GPU (`supports_immad == false`) | out-of-order | `barriers` | `events` |

### Why systolic array GPUs use an in-order queue

`queue_type` is forced to `in_order` whenever oneDNN is used:

```cpp
// execution_config.cpp: ExecutionConfig::finalize_impl
if (!is_set_by_user(ov::intel_gpu::use_onednn) && info.supports_immad) {
    m_use_onednn = true;
}
if (get_use_onednn()) {
    m_queue_type = QueueTypes::in_order;
}
```

`use_onednn` defaults to `true` when `supports_immad` is `true` (i.e. the device has a systolic array).

* Historically, oneDNN did not support out-of-order queues, so in-order was the only option.
* oneDNN can now work with out-of-order queues, but the plugin still forces in-order queue for `supports_immad` devices, since it performs better in practice. Ticket: 105103
* Because the queue is in-order, no explicit synchronization between kernels is required (`SyncMethods::none`) — the driver guarantees execution order.

### Why non-systolic GPUs use barriers instead of events

For `supports_immad == false` devices, the queue stays out-of-order, so dependencies between kernels must be enforced explicitly. Two options exist: per-kernel `cl_event` wait-lists (`events`), or `clEnqueueBarrierWithWaitList` calls (`barriers`).

The plugin picks `barriers` for these devices for performance reason.

Each enqueued command gets a monotonically increasing "queue stamp" (`_queue_counter`). A new barrier is only inserted when a dependency was enqueued after the last barrier, avoiding redundant barriers when a previous one already covers all outstanding work.

### Why `events` is used only when profiling is enabled

Real `cl_event` objects are needed to query per-kernel timing via `clGetEventProfilingInfo` (submission/start/end timestamps). `barriers`/`none` cannot provide since they don't produce event per kernel.

When `events` mode is active, `ocl_stream::enqueue_kernel()` passes the dependency `cl::Event`s directly as the wait-list of `clEnqueueNDRangeKernel`, and every kernel produces its own output event:

```cpp
if (m_sync_method == SyncMethods::events) {
    dep_events = utils::get_cl_events(deps);
    dep_events_ptr = &dep_events;
} else if (m_sync_method == SyncMethods::barriers) {
    sync_events(deps, is_output);
}
...
bool set_output_event = m_sync_method == SyncMethods::events || is_output;
```

## Summary

* `supports_immad == true` (systolic array GPU): in-order queue, `SyncMethods::none` — no explicit sync needed; kept in-order for performance even though oneDNN now supports out-of-order queues (CVS-105103).
* `supports_immad == false`: out-of-order queue, `SyncMethods::barriers` — cheaper than per-kernel events.
* `SyncMethods::events` is used only when profiling is enabled, on either device type, because per-kernel `cl_event`s are required to extract profiling timestamps.

This same `SyncMethods` abstraction is shared by the SYCL ([sycl_stream.cpp](../src/runtime/sycl/sycl_stream.cpp)) and Level Zero ([ze_stream.cpp](../src/runtime/ze/ze_stream.cpp)) backends.
