---
sidebar_position: 2
sidebar_label: "Run Notebooks"
description: "Run Jupyter notebooks to go through the tutorials and learn OpenVINO™ toolkit."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Run Notebooks

This guide will show you how to launch and manage OpenVINO™ Notebooks on your machine.
Before you proceed, make sure you have followed the
[installation steps](notebooks-installation.md).

## Launch a Single Notebook

If you want to launch only one notebook, such as the *Monodepth* notebook, run the
command below.

```text
jupyter lab notebooks/vision-monodepth/vision-monodepth.ipynb
```

## Launch All Notebooks

```text
jupyter lab notebooks
```

In your browser, select a notebook from the file browser in Jupyter Lab, using the left
sidebar. Each tutorial is located in a subdirectory within the `notebooks` directory.

launch-jupyter

## Manage the Notebooks

### Shut Down Jupyter Kernel

To end your Jupyter session, press `Ctrl-c`. This will prompt you to
`Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.

### Deactivate Virtual Environment

First, make sure you use the terminal window where you activated `openvino_env`.
To deactivate your `virtualenv`, simply run:

```text
deactivate
```

This will deactivate your virtual environment.

### Reactivate Virtual Environment

To reactivate your environment, run:

<Tabs>
  <TabItem value="windows" label="Windows">
  ```text
  source openvino_env\Scripts\activate
  ```
  </TabItem>
  <TabItem value="linux" label="Linux">
  ```text
  source openvino_env/bin/activate
  ```
  </TabItem>
  <TabItem value="macos" label="macOS">
  ```text
  source openvino_env/bin/activate
  ```
  </TabItem>
</Tabs>

Then type `jupyter lab` or `jupyter notebook` to launch the notebooks again.

### Delete Virtual Environment

This operation is optional. However, if you want to remove your virtual environment,
simply delete the `openvino_env` directory:

<Tabs>
  <TabItem value="windows" label="Windows">
  ```text
  rmdir /s openvino_env
  ```
  </TabItem>
  <TabItem value="linux" label="Linux">
  ```text
  rm -rf openvino_env
  ```
  </TabItem>
  <TabItem value="macos" label="macOS">
  ```text
  rm -rf openvino_env
  ```
  </TabItem>
</Tabs>

### Remove openvino\_env Kernel from Jupyter

```text
jupyter kernelspec remove openvino_env
```

If you run into issues, check the [Troubleshooting](#-troubleshooting), and
[FAQs](#-faq) sections or start a GitHub
[discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions).

## Additional Resources

- [OpenVINO™ Notebooks - Github Repository Source](https://github.com/openvinotoolkit/openvino_notebooks/)
