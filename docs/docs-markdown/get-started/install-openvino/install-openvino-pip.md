---
sidebar_position: 2
sidebar_label: "Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository"
description: "Learn how to install OpenVINO™ Runtime on Windows, Linux, and macOS operating systems, using a PyPi package."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository

:::note
Note that the PyPi distribution:

- offers the Python API only
- is dedicated to users of all major OSes: Windows, Linux, and macOS
  (all x86\_64 / arm64 architectures)
- macOS offers support only for CPU inference

Before installing OpenVINO, see the
[System Requirements page](../../../about-openvino/release-notes-openvino/system-requirements).
:::

## Installing OpenVINO Runtime

### Step 1. Set Up Python Virtual Environment

Use a virtual environment to avoid dependency conflicts.
To create a virtual environment, use the following command:

<Tabs>
  <TabItem value="windows" label="Windows">
  ```bash
  python -m venv openvino_env
  ```
  </TabItem>
  <TabItem value="linux-and-macos" label="Linux and macOS">
  ```bash
  python3 -m venv openvino_env
  ```
  </TabItem>
</Tabs>

### Step 2. Activate Virtual Environment

<Tabs>
  <TabItem value="windows" label="Windows">
  ```bash
  openvino_env\Scripts\activate
  ```
  </TabItem>
  <TabItem value="linux-and-macos" label="Linux and macOS">
  ```bash
  source openvino_env/bin/activate
  ```
  </TabItem>
</Tabs>

:::info[Important]
The above command must be re-run every time a new command terminal window is opened.
:::

### Step 3. Set Up and Update PIP to the Highest Version

Use the following command:

```bash
python -m pip install --upgrade pip
```

### Step 4. Install the Package

Use the following command to install either the base or GenAI OpenVINO package:

```python
python -m pip install openvino
```

### Step 5. Verify that the Package Is Installed

Run the command below:

```bash
python -c "from openvino import Core; print(Core().available_devices)"
```

If installation was successful, you will see the list of available devices.

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
[list of additional configurations](configurations/index.md)
to see if your case needs any of them.

**Simplified Build and Integration**  
The package includes CMake configurations, precompiled static libraries, and headers, which
can be easily accessed through the Python API. You can use the get\_cmake\_path() method to
retrieve the paths to the CMake configurations and libraries:

```python
from openvino.utils import get_cmake_path
cmake_path = get_cmake_path()
```

For detailed instructions on how to use these configurations in your build setup, check out the
[Create a library with extensions](../../documentation/openvino-extensibility/index.md#create-a-library-with-extensions) section.

## What's Next?

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning
applications! Learn more about how to integrate a model in OpenVINO applications by trying out
the following tutorials.

<div style={{width: '400px'}}>

![](https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif)

</div>

Try the [Python Quick Start Example](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth)
to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside
your web browser.

### Get started with Python

Visit the [Tutorials](../../../get-started/learn-openvino/interactive-tutorials-python) page for more
Jupyter Notebooks to get you started with OpenVINO, such as:

- [OpenVINO Python API Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api)
- [Basic image classification program with Hello Image Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world)
- [Convert a PyTorch model and use it for image background removal](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal)

## Additional Resources

- Intel® Distribution of OpenVINO™ [toolkit home page](https://software.intel.com/en-us/openvino-toolkit)
