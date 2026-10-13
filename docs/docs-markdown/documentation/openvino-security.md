---
sidebar_position: 5
sidebar_label: "OpenVINO Security"
---

# OpenVINO Security

Deploying deep learning models for OpenVINO may raise security and privacy issues.
Trained models are often valuable intellectual property and you may choose to protect them
with encryption or other security tools.

Actual security and privacy requirements depend on your unique deployment scenario.
This section provides general guidance on using OpenVINO tools and libraries securely.
The main security measure for OpenVINO is its
[Security Add-on](../about-openvino/openvino-ecosystem/openvino-project/openvino-security-add-on.md). You can find its description
in the Ecosystem section.


## Using Encrypted Models with OpenVINO {#encrypted-models}

Deploying deep-learning capabilities to edge devices can present security challenges like ensuring
inference integrity, or providing copyright protection of your deep-learning models.

One possible solution is to use cryptography to protect models as they are deployed and stored
on edge devices. Model encryption, decryption and authentication are not provided by OpenVINO
but can be implemented with third-party tools (i.e., OpenSSL). While implementing encryption,
ensure that  the latest versions of tools are used and follow cryptography best practices.

This guide presents how to use OpenVINO securely with protected models.

### Secure Model Deployment

After a model is optimized by model conversion API, it's deployed to target devices in the
OpenVINO Intermediate Representation (OpenVINO IR) format. An optimized model is stored on edge
device and is executed by the OpenVINO Runtime. TensorFlow, TensorFlow Lite, ONNX and PaddlePaddle
models can be read natively by OpenVINO Runtime as well.

Encrypting and optimizing model before deploying it to the edge device can be used to protect
deep-learning models. The edge device should keep the stored model protected all the time
and have the model decrypted **in runtime only** for use by the OpenVINO Runtime.

![](deploy_encrypted_model.svg)

### Loading Encrypted Models

The OpenVINO Runtime requires model decryption before loading. Allocate a temporary memory block
for model decryption and use the `ov::Core::read_model` method to load the model from a memory
buffer. For more information, see the `ov::Core` Class Reference Documentation.

```cpp
std::vector<uint8_t> model_data, weights_data;

std::string password; // taken from an user
std::ifstream model_file("model.xml"), weights_file("model.bin");

// Read model files and decrypt them into temporary memory block
decrypt_file(model_file, password, model_data);
decrypt_file(weights_file, password, weights_data);
```

Hardware-based protection such as Intel Software Guard Extensions (Intel SGX) can be used to protect
decryption operation secrets and bind them to a device. For more information, see
the [Intel Software Guard Extensions](https://software.intel.com/en-us/sgx).

Use the <a href="../api/c_cpp_api/group__ov__dev__exec__model.html#classov_1_1_core_1ae0576a95f841c3a6f5e46e4802716981">ov::Core::read_model</a>
to set model representations and weights respectively.

Currently there is no way to read external weights from memory for ONNX models.
The `ov::Core::read_model(const std::string& model, const Tensor& weights)` method
should be called with `weights` passed as an empty `ov::Tensor`.

```cpp
ov::Core core;
// Load model from temporary memory block
std::string str_model(model_data.begin(), model_data.end());
auto model = core.read_model(str_model,
    ov::Tensor(ov::element::u8, {weights_data.size()}, weights_data.data()));
```

Encrypted models that have already been compiled, in the form of blob files,
can be loaded using the
<a href="../api/c_cpp_api/group__ov__runtime__cpp__api.html#_CPPv4N2ov4Core12import_modelERNSt7istreamERKNSt6stringERK6AnyMap">ov::Core::import_model</a>
method, as shown in the code sample below:

```cpp
ov::Core core;
// Import a model from a blob.
std::ifstream compiled_blob(blob, std::ios_base::in | std::ios_base::binary);
auto compiled_model = core.import_model(compiled_blob, "CPU");
```

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit [home page](https://software.intel.com/en-us/openvino-toolkit).
- [Convert a Model](../openvino-workflow/model-preparation/convert-model-to-ir/index.md).
- [OpenVINO™ Runtime User Guide](../openvino-workflow/running-inference/index.md).
- For more information on Sample Applications, see the [OpenVINO Samples Overview](../get-started/learn-openvino/openvino-samples/index.md)
