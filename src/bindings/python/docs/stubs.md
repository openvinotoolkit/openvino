# Python Stub Files in OpenVINO

## What are stub `.pyi` files?

Stub files (`.pyi`) are used to provide type hints for Python code. They describe the structure of a Python module, including its classes, methods, and functions, without containing any actual implementation. Stub files are particularly useful for improving code readability, enabling better autocompletion in IDEs, and supporting static type checking tools like `mypy`.

## Automation of stub file generation in OpenVINO

In OpenVINO, the generation of stub files is automated as part of the development workflow. This is now handled by a GitHub Actions workflow that runs on a schedule (weekdays at 2:00 AM UTC) and can be triggered manually when needed. The workflow is defined in [.github/workflows/update_pyapi_stubs.yml](../../../../.github/workflows/update_pyapi_stubs.yml). The related Python dependency is `pybind11-stubgen` for stub generation.

### Running stub generation manually

To ensure that `pybind11-stubgen` works correctly, the `openvino` Python package must include your latest changes. This can be achieved by:

- Setting the `PYTHONPATH` environment variable to point to your local OpenVINO build directory. For example:

    ```bash
    export PYTHONPATH=$PYTHONPATH:/home/pwysocki/openvino/bin/intel64/Release/python
    ```

- Installing an up-to-date wheel containing your changes using `pip install`.

Then, to run stub generation manually, ensure your changes are installed and run:
    ```bash
    python openvino/src/bindings/python/scripts/generate_pyapi_stubs.py
    ```

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
