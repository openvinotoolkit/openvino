# Debugging OpenVINO Python API with C++ Breakpoints

This guide explains how to debug the OpenVINO Python API by setting breakpoints in the underlying C++ code. Since the Python API is implemented using pybind11 bindings over C++ code, debugging requires a special setup to step through both Python and C++ layers.

## Prerequisites

### Build OpenVINO with Debug Symbols

To debug C++ code, you need to build OpenVINO with debug symbols enabled. Add the following CMake flags:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DENABLE_PYTHON=ON \
      ...
```

For complete build instructions, refer to [Building the OpenVINO Python API](./build.md).

### Required Tools

- GDB (GNU Debugger) - usually pre-installed on Linux systems
- Python with debug symbols (recommended for better debugging experience)
- VS Code with C/C++ extension (or any other debugger that supports C++ debugging)

### Install Test Dependencies

```bash
pip install -r src/bindings/python/requirements_test.txt
```

## Understanding the Python-C++ Bridge

The OpenVINO Python API is a thin wrapper around C++ implementation. When you call a Python method like `tensor.get_size()`, the execution flow is:

1. Python code calls the method
2. pybind11 directs the call to C++
3. C++ implementation in `ov::Tensor::get_size()` executes
4. Result is directed back to Python

To debug this, we need to:
- Start Python as the main process
- Attach GDB to debug the C++ code
- Set breakpoints in C++ source files

## Debugging Setup in VS Code

### 1. Create or Update `.vscode/launch.json`

Create a launch configuration in your workspace's `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python API Debug",
            "type": "cppdbg",
            "request": "launch",
            "program": "/path/to/your/python",
            "args": [
                "-m", "pytest", 
                "tests/test_runtime/test_tensor.py::test_tensor_from_numpy",
                "-v"
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/src/bindings/python/tests",
            "environment": [
                {
                    "name": "PYTHONPATH",
                    "value": "${workspaceFolder}/bin/intel64/Debug/python"
                },
                {
                    "name": "LD_LIBRARY_PATH",
                    "value": "${workspaceFolder}/bin/intel64/Debug"
                }
            ],
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```

### 2. Configuration Parameters Explained

- **`program`**: Path to your Python interpreter. Use `which python` or `pyenv which python` to find it.
- **`args`**: Arguments passed to Python. Here we're running pytest with a specific test.
- **`cwd`**: Working directory - set to the tests directory.
- **`PYTHONPATH`**: Points to the built Python bindings.
- **`LD_LIBRARY_PATH`**: Points to the built OpenVINO libraries.

### 3. Adjust Paths for Your Environment

Replace the following in the configuration:

- `/path/to/your/python` → Your Python executable path
  - For pyenv: `~/.pyenv/versions/<env_name>/bin/python`
  - For system Python: `/usr/bin/python3`
- `${workspaceFolder}` resolves to your OpenVINO repository root

## Step-by-Step Debugging Example

Let's debug the `ov::Tensor::get_size()` method, which is a stable API unlikely to be removed.

### 1. Set a Breakpoint in C++ Code

Open the C++ source file:
```
src/bindings/python/src/pyopenvino/core/tensor.cpp
```

Find the `get_size` method binding:

```cpp
cls.def("get_size",
        &ov::Tensor::get_size,
        R"(
        Gets Tensor's size as total number of elements.

        :rtype: int
        )");
```

The actual implementation is in the core library. Set a breakpoint at line (`&ov::Tensor::get_size`), or better yet, find the implementation in:

```
src/core/include/openvino/core/tensor.hpp
```

And set a breakpoint in the `get_size()` method implementation.

### 2. Prepare a Test Case

You can use an existing test that calls `get_size()`. For example, `tests/test_runtime/test_tensor.py` contains:

```python
def test_tensor_from_numpy():
    arr = np.array([1, 2, 3])
    ov_tensor = ov.Tensor(arr)
    assert ov_tensor.get_size() == arr.size  # This calls C++ code
```

### 3. Update Launch Configuration

Modify the `args` in your launch configuration to run this specific test:

```json
"args": [
    "-m", "pytest", 
    "tests/test_runtime/test_tensor.py::test_tensor_from_numpy",
    "-v", "-s"
]
```

The `-s` flag disables output capturing, making debugging easier.

### 4. Start Debugging

1. Open the C++ file where you want to set a breakpoint
2. Click in the left margin next to the line number to set a breakpoint
3. Press `F5` or click "Run → Start Debugging"
4. Select "Python API Debug" configuration
5. The debugger will start Python, load the bindings, and stop at your C++ breakpoint

### 5. Debug as Normal

Once stopped at the breakpoint, you can:

- **Step through code**
- **Inspect variables**: Hover over variables or use the Variables pane
- **Evaluate expressions**: Use the Debug Console
- **View call stack**: See both Python and C++ frames in the Call Stack pane

### Debugging Without Tests

You can also debug a standalone Python script:

```json
{
    "name": "Python Script Debug",
    "type": "cppdbg",
    "request": "launch",
    "program": "/path/to/python",
    "args": ["${workspaceFolder}/my_script.py"],
    "cwd": "${workspaceFolder}",
    "environment": [
        {
            "name": "PYTHONPATH",
            "value": "${workspaceFolder}/bin/intel64/Debug/python"
        }
    ],
    "MIMode": "gdb"
}
```

**This method does not use or require `launch.json`.** It uses GDB directly from the command line as an alternative to the VS Code debugging setup.

### Attaching to Running Python Process
You can attach GDB to an already running Python process:

1. Start your Python script with a pause:
   ```python
   import os
   import time
   print(f"PID: {os.getpid()}")
   time.sleep(30)  # Time to attach debugger
   ```

2. In another terminal, attach GDB using the process ID:
   ```bash
   gdb -p <PID>
   ```

3. Set breakpoints and continue execution:
   ```
   (gdb) break ov::Tensor::get_size
   (gdb) continue
   ```

## See also
 * [Building the OpenVINO Python API](./build.md)
 * [How to test OpenVINO Python API?](./test_examples.md)
 * [Contributing to OpenVINO Python API](./contributing.md)
 * [OpenVINO README](../../../../README.md)
 * [OpenVINO bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
