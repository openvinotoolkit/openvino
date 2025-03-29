# Python Stub Files in OpenVINO

## What are stub `.pyi` files?

Stub files (`.pyi`) are used to provide type hints for Python code. They describe the structure of a Python module, including its classes, methods, and functions, without containing any actual implementation. Stub files are particularly useful for improving code readability, enabling better autocompletion in IDEs, and supporting static type checking tools like `mypy`.

## Automation of stub file generation in OpenVINO

In OpenVINO, the generation of stub files is automated as part of the development workflow. When building the Python API for the first time, a Git pre-commit hook is installed into the OpenVINO repository's `.git` directory. The related Python dependencies are `pybind11-stubgen` for stub generation and `pre-commit` for automating git hooks.

### What is a git pre-commit hook?

A Git pre-commit hook is a script that runs automatically before a commit is finalized. It allows developers to enforce specific checks or perform automated tasks, such as code formatting, linting, or file generation, ensuring that the repository maintains a consistent and high-quality state.

### What happens during the pre-commit hook?

When the pre-commit hook is triggered, the following steps occur:

1. **Stub file generation**: The `pybind11-stubgen` tool is executed to generate new `.pyi` stub files for the OpenVINO Python API. This tool uses the Python package `openvino` to extract type information and create the stub files, so it's important that currently installed OpenVINO version contains all changes related to your Pull Request.
2. **Automatic addition to commit**: The newly generated `.pyi` files are automatically staged and added to the current commit. This ensures that the stub files are always up-to-date with the latest changes in the Python API.

### Ensuring changes are installed

To ensure that `pybind11-stubgen` works correctly, the `openvino` Python package must include your latest changes. This can be achieved by:

- Setting the `PYTHONPATH` environment variable to point to your local OpenVINO build directory. For example:

    ```bash
    export PYTHONPATH=$PYTHONPATH:/home/pwysocki/openvino/bin/intel64/Release/python
    ```

- Installing an up-to-date wheel containing your changes using `pip install`.

### Skipping the pre-commit hook

If you need to skip the pre-commit hook for any reason, you can do so by setting the `SKIP` environment variable before running the `git commit` command:

```bash
SKIP=generate_stubs git commit -m "Your commit message"
```

This bypasses the stub file generation process, allowing you to commit without triggering the hook.

### Uninstalling the pre-commit hook

If you wish to remove the hook, you can run:

```bash
pre-commit uninstall
```

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
