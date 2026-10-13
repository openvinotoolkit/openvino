---
sidebar_position: 1
sidebar_label: "Install OpenVINO™ GenAI"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install OpenVINO™ GenAI

OpenVINO GenAI is a tool, simplifying generative AI model inference. It is based on the
OpenVINO Runtime, hiding the complexity of the generation process and minimizing the amount of
code required. You provide a model and the input context directly to the tool, while it
performs tokenization of the input text, executes the generation loop on the selected device,
and returns the generated content. For a quickstart guide, refer to the
[GenAI API Guide](../../openvino-workflow-generative/inference-with-genai).

To see OpenVINO GenAI in action, check these Jupyter notebooks:
[LLM-powered Chatbot](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot)
and
[LLM Instruction-following pipeline](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering).

OpenVINO GenAI is available for installation via PyPI, NPM and Archive distributions.
A [detailed guide](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2026/0/src/docs/BUILD.md)
on how to build OpenVINO GenAI is available in the OpenVINO GenAI repository.

## PyPI Installation

To install the GenAI package via PyPI, follow the standard [installation steps](install-openvino-pip.md),
but use the *openvino-genai* package instead of *openvino*:

```python
python -m pip install openvino-genai
```

## NPM Installation

Install the openvino-genai-node package from [NPM]([https:/www.npmjs.com/package/openvino-genai-node](https:/www.npmjs.com/package/openvino-genai-node)).
The package will automatically install the required openvino-node dependency with the matching version.

```bash
npm install openvino-genai-node
```

:::note
The OpenVINO GenAI Node.js package depends on the OpenVINO Node.js package.
These packages must be of the same version to ensure compatibility.
The npm installation automatically handles this requirement.
:::

## Archive Installation

The OpenVINO GenAI archive package includes the OpenVINO™ Runtime, as well as [Tokenizers](../../openvino-workflow-generative/ov-tokenizers).
It installs the same way as the standard OpenVINO Runtime, so follow its installation steps,
just use the OpenVINO GenAI package instead:

### Linux

<Tabs>
  <TabItem value="x86-64" label="x86_64">
  <Tabs>
    <TabItem value="ubuntu-26" label="Ubuntu 26.04">
    ```bash
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/linux/openvino_genai_ubuntu26_2026.4.1.0_x86_64.tar.gz --output openvino_genai_2026.4.1.0.tgz
    tar -xf openvino_genai_2026.4.1.0.tgz
    ```
    </TabItem>
    <TabItem value="ubuntu-24" label="Ubuntu 24.04">
    ```bash
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/linux/openvino_genai_ubuntu24_2026.4.1.0_x86_64.tar.gz --output openvino_genai_2026.4.1.0.tgz
    tar -xf openvino_genai_2026.4.1.0.tgz
    ```
    </TabItem>
    <TabItem value="ubuntu-22" label="Ubuntu 22.04">
    ```bash
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/linux/openvino_genai_ubuntu22_2026.4.1.0_x86_64.tar.gz --output openvino_genai_2026.4.1.0.tgz
    tar -xf openvino_genai_2026.4.1.0.tgz
    ```
    </TabItem>
    <TabItem value="rhel-9" label="RHEL 9">
    ```bash
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/linux/openvino_genai_rhel9_2026.4.1.0_x86_64.tar.gz --output openvino_genai_2026.4.1.0.tgz
    tar -xf openvino_genai_2026.4.1.0.tgz
    ```
    </TabItem>
    <TabItem value="rhel-8" label="RHEL 8">
    ```bash
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/linux/openvino_genai_rhel8_2026.4.1.0_x86_64.tar.gz --output openvino_genai_2026.4.1.0.tgz
    tar -xf openvino_genai_2026.4.1.0.tgz
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="arm-64" label="ARM 64-bit">
  ```bash
  curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/linux/openvino_genai_ubuntu22_2026.4.1.0_arm64.tar.gz --output openvino_genai_2026.4.1.0.tgz
  tar -xf openvino_genai_2026.4.1.0.tgz
  ```
  </TabItem>
</Tabs>

### Windows

```bash
cd <user_home>/Downloads
curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/windows/openvino_genai_windows_2026.4.1.0_x86_64.zip --output openvino_genai_2026.4.1.0.zip
```

### macOS

<Tabs>
  <TabItem value="arm-64" label="ARM, 64-bit">
  ```bash
  curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/macos/openvino_genai_macos_12_6_2026.4.1.0_arm64.tar.gz --output openvino_genai_2026.4.1.0.tgz
  tar -xf openvino_genai_2026.4.1.0.tgz
  ```
  </TabItem>
</Tabs>

Here are the full guides:
[Linux](install-openvino-archive-linux.md),
[Windows](install-openvino-archive-windows.md), and
[macOS](install-openvino-archive-macos.md).
