---
sidebar_position: 2
sidebar_label: "Install OpenVINO™ Runtime via Homebrew"
description: "Learn how to install OpenVINO™ Runtime on Linux and macOS operating systems, using Homebrew."
---

# Install OpenVINO™ Runtime via Homebrew

:::note
Note that the [Homebrew](https://brew.sh/) distribution:

- offers both C/C++ and Python APIs
- does not offer support for NPU inference
- is dedicated to macOS (both arm64 and x86\_64) and Linux (x86\_64 only) users.

Before installing OpenVINO, see the
[System Requirements page](../../../about-openvino/release-notes-openvino/system-requirements).
:::

:::note
This community-maintained distribution channel is provided to help users explore and evaluate OpenVINO.

Please note that accuracy, performance, and behavior may differ from officially supported OpenVINO distributions, and are not guaranteed by the OpenVINO team in this channel. Due to the community-driven nature of this distribution channel, the OpenVINO team does not guarantee timely updates aligned with official releases, nor update availability for all OpenVINO versions.

For production deployments and product integration, we recommend using officially supported distribution channels (for example, official S3 archives or PyPI packages), which provide validated builds and defined guarantees.
:::

## Installing OpenVINO Runtime

1. Make sure that you have installed Homebrew on your system. If not, follow the instructions on [the Homebrew website](https://brew.sh/) to install and configure it.
2. Run the following command in the terminal:
   
   ```bash
   brew install openvino
   ```
3. Check if the installation was successful by listing all Homebrew packages:
   
   ```bash
   brew list
   ```

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
[list of additional configurations](configurations/index.md)
to see if your case needs any of them.

## Uninstalling OpenVINO

To uninstall OpenVINO via Homebrew, use the following command:

```bash
brew uninstall openvino
```

## What's Next?

Now that you've installed OpenVINO Runtime, you can try the following things:

- Learn more about [OpenVINO Workflow](../../../openvino-workflow).
- To prepare your models for working with OpenVINO, see [Model Preparation](../../../openvino-workflow/model-preparation).
- See pre-trained deep learning models on [Hugging Face](https://huggingface.co/OpenVINO).
- Learn more about [Inference with OpenVINO Runtime](../../../openvino-workflow/running-inference).
- See sample applications in [OpenVINO toolkit Samples Overview](../../../get-started/learn-openvino/openvino-samples).
- Check out the OpenVINO [product home page](https://software.intel.com/en-us/openvino-toolkit).
