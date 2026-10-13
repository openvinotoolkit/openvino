---
sidebar_position: 4
sidebar_label: "Install OpenVINO™ Runtime on Linux From YUM Repository"
description: "Learn how to install OpenVINO™ Runtime on Linux operating system, using the YUM repository."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install OpenVINO™ Runtime on Linux From YUM Repository

:::note
Note that the YUM distribution:

- offers both C/C++ and Python APIs
- is dedicated to Linux users only
- additionally includes code samples

Before installing OpenVINO, see the
[System Requirements page](../../../about-openvino/release-notes-openvino/system-requirements).
:::

## Install OpenVINO Runtime

### Step 1: Set Up the Repository

1. Create a YUM repository file (`openvino.repo`) in the `/tmp` directory as a normal user:
   
   ```bash
   tee > /tmp/openvino.repo << EOF
   [OpenVINO]
   name=Intel(R) Distribution of OpenVINO
   baseurl=https://yum.repos.intel.com/openvino
   enabled=1
   gpgcheck=1
   repo_gpgcheck=1
   gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
   EOF
   ```
2. Move the new `openvino.repo` file to the YUM configuration directory, i.e. `/etc/yum.repos.d`:
   
   ```bash
   sudo mv /tmp/openvino.repo /etc/yum.repos.d
   ```
3. Verify that the new repository is set up properly.
   
   ```bash
   yum repolist | grep -i openvino
   ```
   
   You will see the available list of packages.

To list available OpenVINO packages, use the following command:

```bash
yum list 'openvino*'
```

### Step 2: Install OpenVINO Runtime Using the YUM Package Manager

#### Install OpenVINO Runtime

<Tabs>
  <TabItem value="latest-version" label="The Latest Version">
  Run the following command:

  ```bash
  sudo yum install openvino
  ```
  </TabItem>
  <TabItem value="specific-version" label="A Specific Version">
  Run the following command:

  ```bash
  sudo yum install openvino-<VERSION>.<UPDATE>.<PATCH>
  ```

  For example:

  ```bash
  sudo yum install openvino-2026.4.1
  ```
  </TabItem>
</Tabs>

#### Check for Installed Packages and Version

Run the following command:

```bash
yum list installed 'openvino*'
```

:::note
You can additionally install Python API using one of the alternative methods ([conda](install-openvino-conda.md) or [pip](install-openvino-pip.md)).
:::

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
[list of additional configurations](configurations/index.md)
to see if your case needs any of them.

With the YUM distribution, you can build OpenVINO sample files, as explained in the
[guide for OpenVINO sample applications](../../../get-started/learn-openvino/openvino-samples).
For C++ and C, just run the `build_samples.sh` script:

<Tabs>
  <TabItem value="cpp" label="C++">
  ```bash
  /usr/share/openvino/samples/cpp/build_samples.sh
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```bash
  /usr/share/openvino/samples/c/build_samples.sh
  ```
  </TabItem>
</Tabs>

## Uninstalling OpenVINO Runtime

To uninstall OpenVINO Runtime via YUM, run the following command based on your needs:

<Tabs>
  <TabItem value="latest-version" label="The Latest Version">
  ```bash
  sudo yum autoremove openvino
  ```
  </TabItem>
  <TabItem value="specific-version" label="A Specific Version">
  ```bash
  sudo yum autoremove openvino-<VERSION>.<UPDATE>.<PATCH>
  ```

  For example:

  ```bash
  sudo yum autoremove openvino-2026.4.1
  ```
  </TabItem>
</Tabs>

## What's Next?

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications!
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials:

- Try the [C++ Quick Start Example](../../../get-started/learn-openvino/openvino-samples/get-started-demos)
  for step-by-step instructions on building and running a basic image classification C++ application.
  
  <div style={{width: '400px'}}>

  ![](https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg)

  </div>
- Visit the [Samples](../learn-openvino/openvino-samples/index.md#code-samples) page for other C++ example applications to get you started with OpenVINO, such as:
  
  - [Basic object detection with the Hello Reshape SSD C++ sample](../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd)
  - [Object classification sample](../../../get-started/learn-openvino/openvino-samples/hello-classification)

You can also try the following things:

- Learn more about [OpenVINO Workflow](../../../openvino-workflow).
- To prepare your models for working with OpenVINO, see [Model Preparation](../../../openvino-workflow/model-preparation).
- See pre-trained deep learning models on [Hugging Face](https://huggingface.co/OpenVINO).
- Learn more about [Inference with OpenVINO Runtime](../../../openvino-workflow/running-inference).
- See sample applications in [OpenVINO toolkit Samples Overview](../../../get-started/learn-openvino/openvino-samples).
- Take a glance at the OpenVINO [product home page](https://software.intel.com/en-us/openvino-toolkit) .
