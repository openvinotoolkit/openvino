# Skill: Debug Export or Inference Failure

**Trigger:** User provides an error log or asks to debug a failed export/inference.

## Prerequisites

- Run **optimum_bootstrap** skill first.

## Steps

1. Analyse the error traceback to identify the root cause category:
   - **Unsupported model type** → check if `model_type` exists in `optimum/exporters/openvino/model_configs.py`
   - **Tracing failure** → look for dynamic control flow in the model's `modeling_*.py` file that needs patching
   - **Shape mismatch / IR error** → check dummy input generation in the config class
   - **Missing op** → check OpenVINO op coverage for the PyTorch ops used
   - **OOM** → suggest reducing model size or using weight compression

2. Use `TasksManager.get_supported_tasks_for_model_type()` to confirm existing support:
   ```python
   from optimum.exporters.tasks import TasksManager
   TasksManager.get_supported_tasks_for_model_type("<model_type>", exporter="openvino")
   ```

3. If the model type is unsupported → escalate to **optimum_add_model_support** skill.

4. If a patching issue, read the existing patchers in `model_patcher.py` for similar patterns.

5. Provide a clear diagnosis and, if possible, implement the fix.
