# Get Started with OpenVINO™ Deep Learning Deployment Toolkit (DLDT) on Linux*

This guide provides you with the information that will help you to start using the DLDT on Linux*. With this guide you will learn how to:

1. [Configure the Model Optimizer](#configure-the-model-optimizer)
2. [Prepare a model for sample inference:](#prepare-a-model-for-sample-inference)
   1. [Download a pre-trained model](#download-a-trained-model)
   2. [Convert the model to an Intermediate Representation (IR) with the Model Optimizer](#convert-the-model-to-an-intermediate-representation-with-the-model-optimizer)
3. [Run the Image Classification Sample Application with the model](#run-the-image-classification-sample-application)

## Prerequisites
1. This guide assumes that you have already cloned the `dldt` repo and successfully built the Inference Engine and Samples using the [build instructions](inference-engine/README.md). 
2. The original structure of the repository directories is kept unchanged.

> **NOTE**: Below, the directory to which the `dldt` repository is cloned is referred to as `<DLDT_DIR>`.  

## Configure the Model Optimizer

The Model Optimizer is a Python\*-based command line tool for importing trained models from popular deep learning frameworks such as Caffe\*, TensorFlow\*, Apache MXNet\*, ONNX\* and Kaldi\*.

You cannot perform inference on your trained model without running the model through the Model Optimizer. When you run a pre-trained model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The Intermediate Representation is a pair of files that describe the whole model:

- `.xml`: Describes the network topology
- `.bin`: Contains the weights and biases binary data

For more information about the Model Optimizer, refer to the [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html). 

### Model Optimizer Configuration Steps

You can choose to either configure all supported frameworks at once **OR** configure one framework at a time. Choose the option that best suits your needs. If you see error messages, make sure you installed all dependencies.

> **NOTE**: Since the TensorFlow framework is not officially supported on CentOS*, the Model Optimizer for TensorFlow can't be configured and ran on those systems.  

> **IMPORTANT**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

**Option 1: Configure all supported frameworks at the same time**

1.  Go to the Model Optimizer prerequisites directory:
```sh
cd <DLDT_DIR>/model_optimizer/install_prerequisites
```
2.  Run the script to configure the Model Optimizer for Caffe,
    TensorFlow, MXNet, Kaldi\*, and ONNX:
```sh
sudo ./install_prerequisites.sh
```

**Option 2: Configure each framework separately**

Configure individual frameworks separately **ONLY** if you did not select **Option 1** above.

1.  Go to the Model Optimizer prerequisites directory:
```sh
cd <DLDT_DIR>/model_optimizer/install_prerequisites
```
2.  Run the script for your model framework. You can run more than one script:

   - For **Caffe**:
   ```sh
   sudo ./install_prerequisites_caffe.sh
   ```

   - For **TensorFlow**:
   ```sh
   sudo ./install_prerequisites_tf.sh
   ```

   - For **MXNet**:
   ```sh
   sudo ./install_prerequisites_mxnet.sh
   ```

   - For **ONNX**:
   ```sh
   sudo ./install_prerequisites_onnx.sh
   ```

   - For **Kaldi**:
   ```sh
   sudo ./install_prerequisites_kaldi.sh
   ```
The Model Optimizer is configured for one or more frameworks. Continue to the next session to download and prepare a model for running a sample inference.

## Prepare a Model for Sample Inference

This paragraph contains the steps to get the pre-trained model for sample inference and to prepare the model's optimized Intermediate Representation that Inference Engine uses.

### Download a Trained Model

To run the Image Classification Sample you'll need a pre-trained model to run the inference on. This guide will use the public SqueezeNet 1.1 Caffe* model. You can find and download this model manually or use the OpenVINO™ [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader). 

With the Model Downloader, you can download other popular public deep learning topologies and the [OpenVINO™ pre-trained models](https://github.com/opencv/open_model_zoo/tree/master/intel_models) prepared for running inference for a wide list of inference scenarios: object detection, object recognition, object re-identification, human pose estimation, action recognition and others.

To download the SqueezeNet 1.1 Caffe* model to a models folder with the Model Downloader:
1. Install the [prerequisites](https://github.com/opencv/open_model_zoo/tree/master/model_downloader#prerequisites).
2. Run the `downloader.py` with specifying the topology name and a `<models_dir>` path. For example to download the model to the `~/public_models` directory:
   ```sh
   ./downloader.py --name squeezenet1.1 --output_dir ~/public_models
   ```
   When the model files are successfully downloaded the output similar to the following is printed:
   ```sh
   ###############|| Downloading topologies ||###############

   ========= Downloading /home/username/public_models/classification/squeezenet/1.1/caffe/squeezenet1.1.prototxt
   
   ========= Downloading /home/username/public_models/classification/squeezenet/1.1/caffe/squeezenet1.1.caffemodel
   ... 100%, 4834 KB, 3157 KB/s, 1 seconds passed

   ###############|| Post processing ||###############

   ========= Changing input dimensions in squeezenet1.1.prototxt =========
   ```

### Convert the model to an Intermediate Representation with the Model Optimizer

> **NOTE**: This section assumes that you have configured the Model Optimizer using the instructions from the [Configure the Model Optimizer](#configure-the-model-optimizer) section.

1. Create a `<ir_dir>` directory that will contains the Intermediate Representation (IR) of the model. 

2. Inference Engine can perform inference on a [list of supported devices](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html) using specific device plugins. Different plugins support models of [different precision formats](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html#supported_model_formats), such as FP32, FP16, INT8. To prepare an IR to run inference on a particular hardware, run the Model Optimizer with the appropriate `--data_type` options:

   **For CPU (FP32):**
   ```sh  
   python3 <DLDT_DIR>/model_optimizer/mo.py --input_model <models_dir>/classification/squeezenet/1.1/caffe/squeezenet1.1.caffemodel --data_type FP32 --output_dir <ir_dir>
   ```

   **For GPU and MYRIAD (FP16):**
   ```sh  
   python3 <DLDT_DIR>/model_optimizer/mo.py --input_model <models_dir>/classification/squeezenet/1.1/caffe/squeezenet1.1.caffemodel --data_type FP16 --output_dir <ir_dir>
   ``` 
   After the Model Optimizer script is completed, the produced IR files (`squeezenet1.1.xml`, `squeezenet1.1.bin`) are in the specified `<ir_dir>` directory.

3. Copy the `squeezenet1.1.labels` file from the `<DLDT_DIR>/inference-engine/samples/sample_data/` to the model IR directory. This file contains the classes that ImageNet uses so that the inference results show text instead of classification numbers:
   ```sh
   cp <DLDT_DIR>/inference-engine/samples/sample_data/squeezenet1.1.labels <ir_dir>
   ```

Now you are ready to run the Image Classification Sample Application.

## Run the Image Classification Sample Application

The Inference Engine sample applications are automatically compiled when you built the Inference Engine using the [build instructions](inference-engine/README.md). The binary files are located in the `<DLDT_DIR>/inference-engine/bin/intel64/Release` directory.

Follow the steps below to run the Image Classification sample application on the prepared IR and with an input image: 

1. Go to the samples build directory:
   ```sh
   cd <DLDT_DIR>/inference-engine/bin/intel64/Release
   ```
2. Run the sample executable with specifying the `car.png` file from the `<DLDT_DIR>/inference-engine/samples/sample_data/` directory as an input image, the IR of your model and a plugin for a hardware device to perform inference on:

   **For CPU:**
   ```sh
   ./classification_sample -i <DLDT_DIR>/inference-engine/samples/sample_data/car.png -m <ir_dir>/squeezenet1.1.xml -d CPU
   ```

   **For GPU:**
   ```sh
   ./classification_sample -i <DLDT_DIR>/inference-engine/samples/sample_data/car.png -m <ir_dir>/squeezenet1.1.xml -d GPU
   ```
   
   **For MYRIAD:** 
   >**NOTE**: Running inference on VPU devices (Intel® Movidius™ Neural Compute Stick or Intel® Neural Compute Stick 2) with the MYRIAD plugin requires performing [additional hardware configuration steps](inference-engine/README.md#optional-additional-installation-steps-for-the-intel-movidius-neural-compute-stick-and-neural-compute-stick-2).
   ```sh
   ./classification_sample -i <DLDT_DIR>/inference-engine/samples/sample_data/car.png -m <ir_dir>/squeezenet1.1.xml -d MYRIAD
   ```

When the Sample Application completes, you will have the label and confidence for the top-10 categories printed on the screen. Below is a sample output with inference results on CPU:    
```sh
Top 10 results:

Image /home/user/dldt/inference-engine/samples/sample_data/car.png

classid probability label
------- ----------- -----
817     0.8363345   sports car, sport car
511     0.0946488   convertible
479     0.0419131   car wheel
751     0.0091071   racer, race car, racing car
436     0.0068161   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037564   minivan
586     0.0025741   half track
717     0.0016069   pickup, pickup truck
864     0.0012027   tow truck, tow car, wrecker
581     0.0005882   grille, radiator grille


total inference time: 2.6642941
Average running time of one iteration: 2.6642941 ms

Throughput: 375.3339402 FPS

[ INFO ] Execution successful
```

## Additional Resources

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Inference Engine build instructions](inference-engine/README.md)
* [Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html). 
