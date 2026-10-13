---
sidebar_position: 3
sidebar_label: "Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using APT Repository"
description: "Learn how to install OpenVINO™ Runtime on the Linux operating system, using the APT repository."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using APT Repository

:::note
Note that the APT distribution:

- offers both C/C++ and Python APIs
- is dedicated to Linux users only
- additionally includes code samples

Before installing OpenVINO, see the
[System Requirements page](../../../about-openvino/release-notes-openvino/system-requirements).
:::

## Installing OpenVINO Runtime

### Step 1: Set Up the OpenVINO Toolkit APT Repository

1. Install the GPG key for the repository
   
   1. Download the [GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB](https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB)
      
      You can also use the following command:
      
      ```bash
      wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
      ```
   2. Add this key to the system keyring:
      
      ```bash
      sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
      ```
      
      > **Note:**
      >
      > You might need to install GnuPG:
      >
      > ```bash
      > sudo apt-get install gnupg
      > ```
2. Add the repository via the following command:
   
   <Tabs>
     <TabItem value="ubuntu-22" label="Ubuntu 22">
     ```bash
     echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
     ```
     </TabItem>
     <TabItem value="ubuntu-24" label="Ubuntu 24">
     ```bash
     echo "deb https://apt.repos.intel.com/openvino ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
     ```
     </TabItem>
     <TabItem value="ubuntu-26" label="Ubuntu 26">
     ```bash
     echo "deb https://apt.repos.intel.com/openvino ubuntu26 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
     ```
     </TabItem>
   </Tabs>
3. Update the list of packages via the update command:
   
   ```bash
   sudo apt update
   ```
4. Verify that the APT repository is properly set up. Run the apt-cache command to see a list of all available OpenVINO packages and components:
   
   ```bash
   apt-cache search openvino
   ```

### Step 2: Install OpenVINO Runtime Using the APT Package Manager

1. Install OpenVINO Runtime

<Tabs>
  <TabItem value="latest-version" label="The Latest Version">
  Run the following command:

  ```bash
  sudo apt install openvino
  ```
  </TabItem>
  <TabItem value="specific-version" label="A Specific Version">
  1. Get a list of OpenVINO packages available for installation:
     
     ```bash
     sudo apt-cache search openvino
     ```
  2. Install a specific version of an OpenVINO package:
     
     ```bash
     sudo apt install openvino-<VERSION>.<UPDATE>.<PATCH>
     ```
     
     For example:
     
     ```bash
     sudo apt install openvino-2026.4.1
     ```
  </TabItem>
</Tabs>

:::note
You can use `--no-install-recommends` option to install only required packages.
Keep in mind that the build tools must be installed **separately** if you want to compile the samples.
:::

2. Check for Installed Packages and Versions

Run the following command:

```bash
apt list --installed | grep openvino
```

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
[list of additional configurations](configurations/index.md)
to see if your case needs any of them.

With the APT distribution, you can build OpenVINO sample files, as explained in the
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

Python samples can run as following:

```bash
python3 /usr/share/openvino/samples/python/hello_query_device/hello_query_device.py
```

## Uninstalling OpenVINO Runtime

To uninstall OpenVINO Runtime via APT, run the following command based on your needs:

<Tabs>
  <TabItem value="latest-version" label="The Latest Version">
  ```bash
  sudo apt autoremove openvino
  ```
  </TabItem>
  <TabItem value="specific-version" label="A Specific Version">
  ```bash
  sudo apt autoremove openvino-<VERSION>.<UPDATE>.<PATCH>
  ```

  For example:

  ```bash
  sudo apt autoremove openvino-2026.4.1
  ```
  </TabItem>
</Tabs>

## What's Next?

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications!
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials:

- Try the [C++ Quick Start Example](../../../get-started/learn-openvino/openvino-samples/get-started-demos) for step-by-step
  instructions on building and running a basic image classification C++ application.
  
  <div style={{width: '400px'}}>

  ![](https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg)

  </div>
- Visit the [Samples](../learn-openvino/openvino-samples/index.md#code-samples) page for other C++ example applications to get you started with OpenVINO, such as:
  
  - [Basic object detection with the Hello Reshape SSD C++ sample](../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd)
  - [Object classification sample](../../../get-started/learn-openvino/openvino-samples/hello-classification)

You can also try the following:

- Learn more about [OpenVINO Workflow](../../../openvino-workflow).
- To prepare your models for working with OpenVINO, see [Model Preparation](../../../openvino-workflow/model-preparation).
- See pre-trained deep learning models on [Hugging Face](https://huggingface.co/OpenVINO)
- Learn more about [Inference with OpenVINO Runtime](../../../openvino-workflow/running-inference).
- See sample applications in [OpenVINO toolkit Samples Overview](../../../get-started/learn-openvino/openvino-samples).
- Take a glance at the OpenVINO [product home page](https://software.intel.com/en-us/openvino-toolkit) .
