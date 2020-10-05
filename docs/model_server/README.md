# OpenVINO&trade; Model Server {#openvino_docs_ovms}

OpenVINO&trade; Model Server (OVMS) is a scalable, high-performance solution for serving machine learning models optimized for Intel&reg; architectures. 
The server provides an inference service via gRPC or REST API - making it easy to deploy new algorithms and AI experiments using the same 
architecture as [TensorFlow Serving](https://github.com/tensorflow/serving) for any models trained in a framework that is supported 
by [OpenVINO](https://software.intel.com/en-us/openvino-toolkit). 

The server implements gRPC and REST API framework with data serialization and deserialization using TensorFlow Serving API,
 and OpenVINO&trade; as the inference execution provider. Model repositories may reside on a locally accessible file system (e.g. NFS),
  Google Cloud Storage (GCS), Amazon S3, Minio or Azure Blob Storage.
  
OVMS is now implemented in C++ and provides much higher scalability compared to its predecessor in Python version.
You can take advantage of all the power of Xeon CPU capabilities or AI accelerators and expose it over the network interface.
Read [release notes](https://github.com/openvinotoolkit/model_server/releases) to find out what's new in C++ version.

Review the [Architecture concept](https://github.com/openvinotoolkit/model_server/blob/main/docs/architecture.md) document for more details.

A few key features: 
- Support for multiple frameworks. Serve models trained in popular formats such as Caffe*, TensorFlow*, MXNet* and ONNX*.
- Deploy new [model versions](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#model-version-policy) without changing client code.
- Support for AI accelerators including [Intel Movidius Myriad VPUs](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_VPU.html), 
[GPU](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CL_DNN.html) and [HDDL](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HDDL.html). 
- The server can be enabled both on [Bare Metal Hosts](docs/host.md) or in
[Docker containers](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md).
- [Kubernetes deployments](https://github.com/openvinotoolkit/model_server/blob/main/deploy). The server can be deployed in a Kubernetes cluster allowing the inference service to scale horizontally and ensure high availability.  
- [Model reshaping](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#model-reshaping). The server supports reshaping models in runtime. 
- [Model ensemble](https://github.com/openvinotoolkit/model_server/blob/main/docs/ensemble_scheduler.md) (preview). Connect multiple models to deploy complex processing solutions and reduce overhead of sending data back and forth.

**Note: OVMS has been tested on CentOS* and Ubuntu*. Publically released docker images are based on CentOS.** 

## Build OpenVINO Model Server
Build the docker image using command:
```bash
make docker_build DLDT_PACKAGE_URL=<URL>
```
called from the root directory of the repository.
Note: URL to OpenVINO Toolkit package can be received after registration on [OpenVINO™ Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download).


It will generate the images, tagged as:
* `openvino/model_server:latest` - with CPU, NCS and HDDL support
* `openvino/model_server:latest-gpu` - with CPU, NCS, HDDL and iGPU support

as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.

The release package is compatible with linux machines on which `glibc` version is greater than or equal to the build image version.
For debugging, an image with a suffix `-build` is also generated (i.e. `openvino/model_server-build:latest`).

*Note:* Images include OpenVINO 2021.1 release. <br>


## Run OpenVINO Model Server

A detailed description of how to use OpenVINO Model Server can be found in [ovms_quickstart](https://github.com/openvinotoolkit/model_server/blob/main/docs/ovms_quickstart.md).

More detailed guides to using Model Server in various scenarios can be found here:

* [Models repository configuration](https://github.com/openvinotoolkit/model_server/blob/main/docs/models_repository.md)

* [Using a docker container](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md)

* [Landing on bare metal or virtual machine](https://github.com/openvinotoolkit/model_server/blob/main/docs/host.md)

* [Performance tuning](https://github.com/openvinotoolkit/model_server/blob/main/docs/performance_tuning.md)

* [Model Ensemble Scheduler](https://github.com/openvinotoolkit/model_server/blob/main/docs/ensemble_scheduler.md)


## API documentation

### GRPC 

OpenVINO&trade; Model Server gRPC API is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r2.2/tensorflow_serving/apis). 

**Note:** The implementations for `Predict`, `GetModelMetadata` and `GetModelStatus` function calls are currently available. 
These are the most generic function calls and should address most of the usage scenarios.

[Predict proto](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/predict.proto) defines two message specifications: `PredictRequest` and `PredictResponse` used while calling Prediction endpoint.  
* `PredictRequest` specifies information about the model spec (name and version) and a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/core/framework/tensor.proto) to a string format.
* `PredictResponse` includes a map of outputs serialized by 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/core/framework/tensor.proto) and information about the used model spec.
 
[Get Model Metadata proto](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/get_model_metadata.proto) defines three message definitions used while calling Metadata endpoint:
 `SignatureDefMap`, `GetModelMetadataRequest`, `GetModelMetadataResponse`.

 A function call `GetModelMetadata` accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.

[Get Model Status proto](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/get_model_status.proto) defines three message definitions used while calling Status endpoint:
 `GetModelStatusRequest`, `ModelVersionStatus`, `GetModelStatusResponse` that are used to report all exposed versions including their state in their lifecycle. 

Refer to the [example client code](https://github.com/openvinotoolkit/model_server/blob/main/example_client) to learn how to use this API and submit the requests using the gRPC interface.

Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. It allows you to achieve lower latency, especially with larger input messages like images. 

### REST

OpenVINO&trade; Model Server RESTful API follows the documentation from [tensorflow serving rest api](https://www.tensorflow.org/tfx/serving/api_rest).

Both row and column format of the requests are implemented.

**Note:** Just like with gRPC, only the implementations for `Predict`, `GetModelMetadata` and `GetModelStatus` function calls are currently available. 

Only the numerical data types are supported. 

Review the exemplary clients below to find out more how to connect and run inference requests.

REST API is recommended when the primary goal is in reducing the number of client side python dependencies and simpler application code.


## Known Limitations

* Currently, `Predict`, `GetModelMetadata` and `GetModelStatus` calls are implemented using Tensorflow Serving API. 
* `Classify`, `Regress` and `MultiInference` are not included.
* Output_filter is not effective in the Predict call. All outputs defined in the model are returned to the clients. 


## OpenVINO Model Server Contribution Policy

* All contributed code must be compatible with the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

* All changes needs to have pass linter, unit and functional tests.

* All new features need to be covered by tests.


## References

* [OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

* [TensorFlow Serving](https://github.com/tensorflow/serving)

* [gRPC](https://grpc.io/)

* [RESTful API](https://restfulapi.net/)

* [Inference at scale in Kubernetes](https://www.intel.ai/inference-at-scale-in-kubernetes)

* [OpenVINO Model Server boosts AI](https://www.intel.ai/openvino-model-server-boosts-ai-inference-operations/)


---
\* Other names and brands may be claimed as the property of others.