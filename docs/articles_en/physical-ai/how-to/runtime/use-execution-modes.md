# Use Execution Modes

> **Preview:** `PolicyRuntime` and execution modes are planned APIs. The examples below document the target design.

The `Execution` component decides where inference runs and how requests are scheduled.

## Synchronous

This mode runs inference in the runtime thread.

```yaml
execution:
  class_path: physicalai.runtime.SyncExecution
  init_args:
    mode: chunk
```

This mode is appropriate for simple deployments and debugging.

## Thread Worker

This mode runs inference in a background thread.

```yaml
execution:
  class_path: physicalai.runtime.AsyncExecution
  init_args:
    transport: thread
```

Use this mode when model latency should not block robot timing. Since inference backends typically release the GIL, thread-based execution works well for most use cases.

## Remote

This mode sends inference requests to a policy server.

```yaml
execution:
  class_path: physicalai.runtime.RemoteExecution
  init_args:
    endpoint: http://robot-server:8080
```

Use this mode when the robot host should not hold policy weights or accelerator dependencies.
