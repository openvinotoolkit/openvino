# Using Encrypted Models with OpenVINO&trade;  {#openvino_docs_IE_DG_protecting_model_guide}

Deploying deep-learning capabilities to edge devices can present security
challenges. For example, ensuring inference integrity or providing copyright
protection of your deep-learning models.

One possible solution is to use cryptography to protect models as they are
deployed and stored on edge devices. Model encryption, decryption and
authentication are not provided by OpenVINO&trade; but can be implemented with
third-party tools, like OpenSSL\*. While implementing encryption, ensure that
you use the latest versions of tools and follow cryptography best practices.

This guide demonstrates how to use OpenVINO securely with protected models.

## Secure Model Deployment

After a model is optimized by the OpenVINO Model Optimizer, it's then deployed
to target devices in the Intermediate Representation (IR) format. An optimized
model is stored on an edge device and executed by the Inference Engine.

To protect deep-learning models, you can encrypt an optimized model before
deploying it to the edge device. The edge device should keep the stored model
protected at all times and have the model decrypted **in runtime only** for use
by the Inference Engine.

![deploy_encrypted_model]

## Loading Encrypted Models

The OpenVINO Inference Engine requires model decryption before loading. Allocate
a temporary memory block for model decryption, and use
`InferenceEngine::Core::ReadNetwork` method to load the model from memory buffer.
For more information, see the `InferenceEngine::Core` Class
Reference Documentation.

@snippet openvino/docs/snippets/protecting_model_guide.cpp part0

Hardware-based protection, such as Intel&reg; Software Guard Extensions
(Intel&reg; SGX), can be utilized to protect decryption operation secrets and
bind them to a device. For more information, go to [Intel&reg; Software Guard
Extensions](https://software.intel.com/en-us/sgx).

Use `InferenceEngine::Core::ReadNetwork()` to set model representations and
weights respectively.

@snippet openvino/docs/snippets/protecting_model_guide.cpp part1

[deploy_encrypted_model]: img/deploy_encrypted_model.png

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
- Model Optimizer Developer Guide: [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
- Inference Engine Developer Guide: [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
- For more information on Sample Applications, see the [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html)
- For information on a set of pre-trained models, see the [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_intel_index)
- For information on Inference Engine Tutorials, see the [Inference Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic)
- For IoT Libraries and Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
