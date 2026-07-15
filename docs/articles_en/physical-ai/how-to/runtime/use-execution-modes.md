# Use Execution Modes

The `Execution` component decides where inference runs and how requests are scheduled.

## Synchronous

This mode runs inference in the runtime thread.

```python
from physicalai.runtime import SyncExecution

execution = SyncExecution()
```

This mode is appropriate for simple deployments and debugging.

## Thread Worker

This mode runs inference in a background thread.

```python
from physicalai.runtime import AsyncExecution

execution = AsyncExecution(fps=30)
```

Use this mode when model latency should not block robot timing. Since inference backends typically release the GIL, thread-based execution works well for most use cases.

## Remote

> **Preview:** `RemoteExecution` is not yet implemented.

This mode sends inference requests to a policy server.

```yaml
execution:
  class_path: physicalai.runtime.RemoteExecution
  init_args:
    endpoint: http://robot-server:8080
```

Use this mode when the robot host should not hold policy weights or accelerator dependencies.
