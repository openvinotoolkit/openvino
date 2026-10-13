---
sidebar_position: 4
sidebar_label: "Install OpenVINO™ Runtime via vcpkg"
description: "Learn how to install OpenVINO™ Runtime on Windows, Linux, and macOS operating systems, using vcpkg."
---

# Install OpenVINO™ Runtime via vcpkg

:::note
Note that the vcpkg distribution:

- offers C/C++ API only
- is dedicated to users of all major OSes: Windows, Linux, and macOS
  (all x86\_64 / arm64 architectures)

Before installing OpenVINO, see the
[System Requirements page](../../../about-openvino/release-notes-openvino/system-requirements).
:::

:::note
This community-maintained distribution channel is provided to help users explore and evaluate OpenVINO.

Please note that accuracy, performance, and behavior may differ from officially supported OpenVINO distributions, and are not guaranteed by the OpenVINO team in this channel. Due to the community-driven nature of this distribution channel, the OpenVINO team does not guarantee timely updates aligned with official releases, nor update availability for all OpenVINO versions.

For production deployments and product integration, we recommend using officially supported distribution channels (for example, official S3 archives or PyPI packages), which provide validated builds and defined guarantees.
:::

## Installing OpenVINO Runtime

1. Make sure that you have installed vcpkg on your system. If not, follow the
   [vcpkg installation instructions](https://vcpkg.io/en/getting-started).
2. Install OpenVINO using the following terminal command:
   
   ```bash
   vcpkg install openvino
   ```
   
   vcpkg also enables you to install only selected components, by specifying them in the command.
   See the list of [available features](https://vcpkg.link/ports/openvino), for example:
   
   ```bash
   vcpkg install 'openvino[core,cpu,ir]'
   ```
   
   vcpkg also provides a way to install OpenVINO for any specific configuration you want via [triplets](https://learn.microsoft.com/en-us/vcpkg/users/triplets), for example to install OpenVINO statically on Windows, use:
   
   ```bash
   vcpkg install 'openvino:x64-windows-static'
   ```

Note that the vcpkg installation means building all packages and dependencies from source,
which means the compiler stage will require additional time to complete the process.

After installation, you can use OpenVINO in your product's cmake scripts:

```bash
find_package(OpenVINO REQUIRED)
```

And running from terminal:

```bash
cmake -B <build dir> -S <source dir> -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake
```

Congratulations! You've just Installed and used OpenVINO in your project! For some use cases you may still
need to install additional components. Check the
[list of additional configurations](configurations/index.md)
to see if your case needs any of them.

## Uninstalling OpenVINO

To uninstall OpenVINO via vcpkg, use the following command:

```bash
vcpkg remove openvino
```

## What's Next?

Now that you've installed OpenVINO Runtime, you can try the following things:

- Learn more about [OpenVINO Workflow](../../../openvino-workflow).
- To prepare your models for working with OpenVINO, see [Model Preparation](../../../openvino-workflow/model-preparation).
- See pre-trained deep learning models on [Hugging Face](https://huggingface.co/OpenVINO).
- Learn more about [Inference with OpenVINO Runtime](../../../openvino-workflow/running-inference).
- See sample applications in [OpenVINO toolkit Samples Overview](../../../get-started/learn-openvino/openvino-samples).
- Check out the OpenVINO [product home page](https://software.intel.com/en-us/openvino-toolkit) .
