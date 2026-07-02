# Add Runtime Callbacks

Use callbacks when you need product-specific behavior around the runtime loop.

The following example records observations and actions.

```python
class RecordingCallback:
    def before_send_action(self, *, action, step):
        recorder.write_policy_action(step, action)
        return action

    def on_action_sent(self, *, action, step):
        recorder.write_sent_action(step, action)

    def on_hold(self, *, step, holds):
        recorder.write_hold(step, holds)
```

Attach the callback when you construct the runtime.

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=execution,
    fps=30,
    callbacks=[RecordingCallback()],
)
```

As a general rule, keep workflow-specific logic in callbacks unless the same behavior becomes a reusable runtime primitive.
