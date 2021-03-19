# OpenVINO&trade; Model Server {#openvino_docs_ovms}

![OVMS](https://github.com/openvinotoolkit/model_server/raw/main/docs/ovms.png)

OpenVINO&trade; Model Server (OVMS) is a scalable, high-performance solution for serving machine learning models optimized for Intel&reg; architectures. 
The server provides an inference service via gRPC or REST API - making it easy to deploy new algorithms and AI experiments using the same 
architecture as [TensorFlow* Serving](https://github.com/tensorflow/serving) for any models trained in a framework that is supported 
by [OpenVINO](https://software.intel.com/en-us/openvino-toolkit). 

The server implements gRPC and REST API framework with data serialization and deserialization using TensorFlow Serving API,
 and OpenVINO&trade; as the inference execution provider. Model repositories may reside on a locally accessible file system (for example, NFS),
  Google Cloud Storage\* (GCS), Amazon S3\*, MinIO\*, or Azure Blob Storage\*.
  
OVMS is now implemented in C++ and provides much higher scalability compared to its predecessor in the Python version.
You can take advantage of all the power of Xeon® CPU capabilities or AI accelerators and expose it over the network interface.
Read the [release notes](https://github.com/openvinotoolkit/model_server/releases) to find out what's new in the C++ version.

Review the [Architecture Concept](https://github.com/openvinotoolkit/model_server/blob/main/docs/architecture.md) document for more details.

A few key features: 
- Support for multiple frameworks. Serve models trained in popular formats such as Caffe\*, TensorFlow\*, MXNet\*, and ONNX*.
- Deploy new [model versions](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#model-version-policy) without changing client code.
- Support for AI accelerators including [Intel Movidius Myriad VPUs](../IE_DG/supported_plugins/VPU), 
[GPU](../IE_DG/supported_plugins/CL_DNN), and [HDDL](../IE_DG/supported_plugins/HDDL). 
- The server can be enabled both on [Bare Metal Hosts](https://github.com/openvinotoolkit/model_server/blob/main/docs/host.md) or in
[Docker* containers](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md).
- [Model reshaping](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#model-reshaping). The server supports reshaping models in runtime. 
- [Directed Acyclic Graph scheduler](https://github.com/openvinotoolkit/model_server/blob/main/docs/dag_scheduler.md). Connect multiple models to deploy complex processing solutions and reduce overhead of sending data back and forth.
- [Support for stateful models](https://github.com/openvinotoolkit/model_server/blob/main/docs/stateful_models.md). 


> **NOTE**: Publicly released [Docker images](https://hub.docker.com/r/openvino/model_server) are based on CentOS.


## Run OpenVINO Model Server

Find a detailed description of how to use the OpenVINO Model Server in the [OVMS Quick Start Guide](https://github.com/openvinotoolkit/model_server/blob/main/docs/ovms_quickstart.md).


For more detailed guides on using the Model Server in various scenarios, visit the links below:

* [Models repository configuration](https://github.com/openvinotoolkit/model_server/blob/main/docs/models_repository.md)

* [Using a Docker container](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md)

* [Landing on bare metal or virtual machine](https://github.com/openvinotoolkit/model_server/blob/main/docs/host.md)

* [Performance tuning](https://github.com/openvinotoolkit/model_server/blob/main/docs/performance_tuning.md)

* [Directed Acyclic Graph Scheduler](https://github.com/openvinotoolkit/model_server/blob/main/docs/dag_scheduler.md)

* [Stateful models](https://github.com/openvinotoolkit/model_server/blob/main/docs/stateful_models.md)

* [Helm chart](https://github.com/openvinotoolkit/model_server/tree/main/deploy) or [Kubernetes Operator](https://operatorhub.io/operator/ovms-operator)


## API Documentation

### GRPC 

OpenVINO&trade; Model Server gRPC API is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r2.2/tensorflow_serving/apis). 

> **NOTE:** The implementations for `Predict`, `GetModelMetadata`, and `GetModelStatus` function calls are currently available. 
> These are the most generic function calls and should address most of the usage scenarios.

[Predict proto](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/predict.proto) defines two message specifications: `PredictRequest` and `PredictResponse` used while calling Prediction endpoint.  
* `PredictRequest` specifies information about the model spec, that is name and version, and a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/core/framework/tensor.proto) to a string format.
* `PredictResponse` includes a map of outputs serialized by 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/core/framework/tensor.proto) and information about the used model spec.
 
[Get Model Metadata proto](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/get_model_metadata.proto) defines three message definitions used while calling Metadata endpoint:
 `SignatureDefMap`, `GetModelMetadataRequest`, `GetModelMetadataResponse`.

 A function call `GetModelMetadata` accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.

[Get Model Status proto](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/get_model_status.proto) defines three message definitions used while calling Status endpoint:
 `GetModelStatusRequest`, `ModelVersionStatus`, `GetModelStatusResponse` that report all exposed versions including their state in their lifecycle. 

Refer to the [example client code](https://github.com/openvinotoolkit/model_server/blob/main/example_client) to learn how to use this API and submit the requests using the gRPC interface.

Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. It enables you to achieve lower latency, especially with larger input messages like images. 

### REST

OpenVINO&trade; Model Server RESTful API is compatible with [TensorFlow Serving REST API](https://www.tensorflow.org/tfx/serving/api_rest) for 
functions `Predict`, `GetModelMetadata`, and `GetModelStatus`.

Only the numerical data types are supported both in row and column format.

The REST API is extended with `Config` function to update the configuration and query the list served models.

REST API is recommended when the primary goal is in reducing the number of client side Python dependencies and simpler application code.

[Learn more about using the REST API](https://github.com/openvinotoolkit/model_server/blob/develop/docs/model_server_rest_api.md)

## References

* [Speed and Scale AI Inference Operations Across Multiple Architectures - webinar recording](https://techdecoded.intel.io/essentials/speed-and-scale-ai-inference-operations-across-multiple-architectures/)

* [Whats New Openvino Model Server C++](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/whats-new-openvino-model-server.html)


---
\* Other names and brands may be claimed as the property of others.
