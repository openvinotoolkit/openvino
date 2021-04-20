# OpenVINO&trade; Model Server {#openvino_docs_ovms}

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
- [Kubernetes deployments](https://github.com/openvinotoolkit/model_server/blob/main/deploy). The server can be deployed in a Kubernetes cluster allowing the inference service to scale horizontally and ensure high availability.  
- [Model reshaping](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#model-reshaping). The server supports reshaping models in runtime. 
- [Model ensemble](https://github.com/openvinotoolkit/model_server/blob/main/docs/ensemble_scheduler.md) (preview). Connect multiple models to deploy complex processing solutions and reduce overhead of sending data back and forth.

> **NOTE**: OVMS has been tested on CentOS\* and Ubuntu\*. Publicly released [Docker images](https://hub.docker.com/r/openvino/model_server) are based on CentOS.

## Build OpenVINO Model Server

1. Go to the root directory of the repository.

2. Build the Docker image with the command below:
```bash
make docker_build
```

The command generates:
* Image tagged as `openvino/model_server:latest` with CPU, NCS, and HDDL support
* Image tagged as `openvino/model_server:latest-gpu` with CPU, NCS, HDDL, and iGPU support
* `.tar.gz` release package with OVMS binary and necessary libraries in the `./dist` directory.

The release package is compatible with Linux machines on which `glibc` version is greater than or equal to the build image version.
For debugging, the command also generates an image with a suffix `-build`, namely `openvino/model_server-build:latest`.

> **NOTE**: Images include OpenVINO 2021.1 release.


## Run OpenVINO Model Server

Find a detailed description of how to use the OpenVINO Model Server in the [OVMS Quick Start Guide](https://github.com/openvinotoolkit/model_server/blob/main/docs/ovms_quickstart.md).


For more detailed guides on using the Model Server in various scenarios, visit the links below:

* [Models repository configuration](https://github.com/openvinotoolkit/model_server/blob/main/docs/models_repository.md)

* [Using a Docker container](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md)

* [Landing on bare metal or virtual machine](https://github.com/openvinotoolkit/model_server/blob/main/docs/host.md)

* [Performance tuning](https://github.com/openvinotoolkit/model_server/blob/main/docs/performance_tuning.md)

* [Model Ensemble Scheduler](https://github.com/openvinotoolkit/model_server/blob/main/docs/ensemble_scheduler.md)


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

OpenVINO&trade; Model Server RESTful API follows the documentation from the [TensorFlow Serving REST API](https://www.tensorflow.org/tfx/serving/api_rest).

Both row and column format of the requests are implemented.

> **NOTE**: Just like with gRPC, only the implementations for `Predict`, `GetModelMetadata`, and `GetModelStatus` function calls are currently available. 

Only the numerical data types are supported. 

Review the exemplary clients below to find out more how to connect and run inference requests.

REST API is recommended when the primary goal is in reducing the number of client side Python dependencies and simpler application code.


## Known Limitations

* Currently, `Predict`, `GetModelMetadata`, and `GetModelStatus` calls are implemented using the TensorFlow Serving API. 
* `Classify`, `Regress`, and `MultiInference` are not included.
* `Output_filter` is not effective in the `Predict` call. All outputs defined in the model are returned to the clients. 

## OpenVINO Model Server Contribution Policy

* All contributed code must be compatible with the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

* All changes have to pass linter, unit, and functional tests.

* All new features need to be covered by tests.


## References

* [Speed and Scale AI Inference Operations Across Multiple Architectures - webinar recording](https://techdecoded.intel.io/essentials/speed-and-scale-ai-inference-operations-across-multiple-architectures/)

* [OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

* [TensorFlow Serving](https://github.com/tensorflow/serving)

* [gRPC](https://grpc.io/)

* [RESTful API](https://restfulapi.net/)

* [Inference at Scale in Kubernetes](https://www.intel.ai/inference-at-scale-in-kubernetes)



---
\* Other names and brands may be claimed as the property of others.
