INT8 Quantization with Post-training Optimization Tool (POT) in Simplified Mode tutorial
========================================================================================

This tutorial shows how to quantize a
`ResNet20 <https://github.com/chenyaofo/pytorch-cifar-models>`__ image
classification model, trained on
`CIFAR10 <http://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html>`__
dataset, using the Post-Training Optimization Tool (POT) in Simplified
Mode.

Simplified Mode is designed to make the data preparation step easier,
before model optimization. The mode is represented by an implementation
of the engine interface in the POT API in OpenVINO™. It enables reading
data from an arbitrary folder specified by the user. Currently,
Simplified Mode is available only for image data in PNG or JPEG formats,
stored in a single folder.

   **NOTE:** This mode cannot be used with the accuracy-aware method. It
   is not possible to control accuracy after optimization using this
   mode. However, Simplified Mode can be useful for estimating
   performance improvements when optimizing models.

This tutorial includes the following steps:

-  Downloading and saving the CIFAR10 dataset.
-  Preparing the model for quantization.
-  Compressing the prepared model.
-  Measuring and comparing the performance of the original and quantized
   models.
-  Demonstrating the use of the quantized model for image
   classification.

.. code:: ipython3

    import os
    from pathlib import Path
    import warnings
    
    import torch
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR10
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    from openvino.runtime import Core, Tensor
    
    warnings.filterwarnings("ignore")
    
    # Set the data and model directories
    MODEL_DIR = 'model'
    CALIB_DIR = 'calib'
    CIFAR_DIR = '../data/datasets/cifar10'
    CALIB_SET_SIZE = 300
    MODEL_NAME = 'resnet20'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)

Prepare the calibration dataset
-------------------------------

The following steps are required to prepare the calibration dataset: -
Download the CIFAR10 dataset from `Torchvision.datasets
repository <https://pytorch.org/vision/stable/datasets.html>`__. - Save
the selected number of elements from this dataset as ``.png`` images in
a separate folder.

.. code:: ipython3

    transform = T.Compose([T.ToTensor()])
    dataset = CIFAR10(root=CIFAR_DIR, train=False, transform=transform, download=True)


.. parsed-literal::

    Files already downloaded and verified


.. code:: ipython3

    pil_converter = T.ToPILImage(mode="RGB")
    
    for idx, info in enumerate(dataset):
        im = info[0]
        if idx >= CALIB_SET_SIZE:
            break
        label = info[1]
        pil_converter(im.squeeze(0)).save(Path(CALIB_DIR) / f'{label}_{idx}.png')

Prepare the Model
-----------------

Model preparation includes the following steps: - Download PyTorch model
from Torchvision repository. - Convert the model to ONNX format. - Run
Model Optimizer to convert ONNX to OpenVINO Intermediate Representation
(OpenVINO IR).

.. code:: ipython3

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, skip_validation=True)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    onnx_model_path = Path(MODEL_DIR) / '{}.onnx'.format(MODEL_NAME)
    ir_model_xml = onnx_model_path.with_suffix('.xml')
    ir_model_bin = onnx_model_path.with_suffix('.bin')
    
    torch.onnx.export(model, dummy_input, onnx_model_path)


.. parsed-literal::

    Using cache found in /opt/home/k8sworker/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master


Now, convert this model into the OpenVINO IR using Model Optimizer:

.. code:: ipython3

    !mo -m $onnx_model_path  --output_dir $MODEL_DIR


.. parsed-literal::

    Check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2023_bu_IOTG_OpenVINO-2022-3&content=upg_all&medium=organic or on https://github.com/openvinotoolkit/openvino
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/114-quantization-simplified-mode/model/resnet20.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/114-quantization-simplified-mode/model/resnet20.bin


Compression stage
-----------------

Compress the model with the following command:

``pot -q default -m <path_to_xml> -w <path_to_bin> --engine simplified --data-source <path_to_data>``

.. code:: ipython3

    !pot -q default -m $ir_model_xml -w $ir_model_bin --engine simplified --data-source $CALIB_DIR --output-dir compressed --direct-dump --name $MODEL_NAME


.. parsed-literal::

    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/offline_transformations/__init__.py:10: FutureWarning: The module is private and following namespace `offline_transformations` will be removed in the future, use `openvino.runtime.passes` instead!
      warnings.warn(
    INFO:openvino.tools.pot.app.run:Output log dir: compressed
    INFO:openvino.tools.pot.app.run:Creating pipeline:
     Algorithm: DefaultQuantization
     Parameters:
    	preset                     : performance
    	stat_subset_size           : 300
    	target_device              : ANY
    	model_type                 : None
    	dump_intermediate_model    : False
    	inplace_statistics         : True
    	exec_log_dir               : compressed
     ===========================================================================
    INFO:openvino.tools.pot.data_loaders.image_loader:Layout value is set [N,C,H,W]
    INFO:openvino.tools.pot.pipeline.pipeline:Inference Engine version:                2022.3.0-9052-9752fafe8eb-releases/2022/3
    INFO:openvino.tools.pot.pipeline.pipeline:Model Optimizer version:                 2022.3.0-9052-9752fafe8eb-releases/2022/3
    INFO:openvino.tools.pot.pipeline.pipeline:Post-Training Optimization Tool version: 2022.3.0-9052-9752fafe8eb-releases/2022/3
    INFO:openvino.tools.pot.statistics.collector:Start computing statistics for algorithms : DefaultQuantization
    INFO:openvino.tools.pot.statistics.collector:Computing statistics finished
    INFO:openvino.tools.pot.pipeline.pipeline:Start algorithm: DefaultQuantization
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Start computing statistics for algorithm : ActivationChannelAlignment
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Computing statistics finished
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Start computing statistics for algorithms : MinMaxQuantization,FastBiasCorrection
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Computing statistics finished
    INFO:openvino.tools.pot.pipeline.pipeline:Finished: DefaultQuantization
     ===========================================================================


Compare Performance of the Original and Quantized Models
--------------------------------------------------------

Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
- an inference performance measurement tool in OpenVINO.

   **NOTE**: For more accurate performance, it is recommended to run
   benchmark_app in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change CPU to GPU to benchmark
   on GPU. Run ``benchmark_app --help`` to see an overview of all
   command-line options.

.. code:: ipython3

    optimized_model_path = Path('compressed/optimized')
    optimized_model_xml = optimized_model_path / '{}.xml'.format(MODEL_NAME)
    optimized_model_bin = optimized_model_path / '{}.bin'.format(MODEL_NAME)

.. code:: ipython3

    # Inference FP32 model (OpenVINO IR)
    !benchmark_app -m $ir_model_xml -d CPU -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 6.18 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input.1 (node: input.1) : f32 / [...] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     208 (node: 208) : f32 / [...] / [1,10]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input.1 (node: input.1) : u8 / [N,C,H,W] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     208 (node: 208) : f32 / [...] / [1,10]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 75.31 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input.1'!. This input will be filled with random values!
    [ INFO ] Fill input 'input.1' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 1.08 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            969060 iterations
    [ INFO ] Duration:         60000.75 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.69 ms
    [ INFO ]    Average:       0.71 ms
    [ INFO ]    Min:           0.41 ms
    [ INFO ]    Max:           12.50 ms
    [ INFO ] Throughput:   16150.80 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    !benchmark_app -m $optimized_model_xml -d CPU -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 9.50 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input.1 (node: input.1) : f32 / [...] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     208 (node: 208) : f32 / [...] / [1,10]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input.1 (node: input.1) : u8 / [N,C,H,W] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     208 (node: 208) : f32 / [...] / [1,10]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 117.27 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input.1'!. This input will be filled with random values!
    [ INFO ] Fill input 'input.1' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 0.68 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            1587024 iterations
    [ INFO ] Duration:         60000.59 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.35 ms
    [ INFO ]    Average:       0.36 ms
    [ INFO ]    Min:           0.22 ms
    [ INFO ]    Max:           13.28 ms
    [ INFO ] Throughput:   26450.14 FPS


Demonstration of the results
----------------------------

This section demonstrates how to use the compressed model by running the
optimized model on a subset of images from the CIFAR10 dataset and shows
predictions, using the model.

The first step is to load the model:

.. code:: ipython3

    ie = Core()
    
    compiled_model = ie.compile_model(str(optimized_model_xml))

.. code:: ipython3

    # Define all possible labels from the CIFAR10 dataset.
    labels_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    all_images = []
    all_labels = []
    
    # Get all images and their labels. 
    for batch in dataset:
        all_images.append(torch.unsqueeze(batch[0], 0))
        all_labels.append(batch[1])

The code below defines the function that shows the images and their
labels, using the indexes and two lists created in the previous step:

.. code:: ipython3

    def plot_pictures(indexes: list, images=all_images, labels=all_labels):
        """Plot images with the specified indexes.
        :param indexes: a list of indexes of images to be displayed.
        :param images: a list of images from the dataset.
        :param labels: a list of labels for each image.
        """
        num_pics = len(indexes)
        _, axarr = plt.subplots(1, num_pics)
        for idx, im_idx in enumerate(indexes):
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            pic = np.rollaxis(images[im_idx].squeeze().numpy(), 0, 3)
            axarr[idx].imshow(pic)
            axarr[idx].set_title(labels_names[labels[im_idx]])

Use the code below, to define a function that uses the optimized model
to obtain predictions for the selected images:

.. code:: ipython3

    def infer_on_images(net, indexes: list, images=all_images):
        """ Inference model on a set of images.
        :param net: model on which do inference
        :param indexes: a list of indexes of images to infer on.
        :param images: a list of images from the dataset.
        """
        predicted_labels = []
        infer_request = net.create_infer_request()
        for idx in indexes:
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            input_tensor = Tensor(array=images[idx].detach().numpy(), shared_memory=True)
            infer_request.set_input_tensor(input_tensor)
            infer_request.start_async()
            infer_request.wait()
            output = infer_request.get_output_tensor()
            result = list(output.data)
            result = labels_names[np.argmax(result[0])]
            predicted_labels.append(result)
        return predicted_labels

.. code:: ipython3

    indexes_to_infer = [0, 1, 2]  # to plot specify indexes
    
    plot_pictures(indexes_to_infer)
    
    results_quanized = infer_on_images(compiled_model, indexes_to_infer)
    
    print(f"Image labels using the quantized model : {results_quanized}.")


.. parsed-literal::

    Image labels using the quantized model : ['cat', 'ship', 'ship'].



.. image:: 114-quantization-simplified-mode-with-output_files/114-quantization-simplified-mode-with-output_22_1.png

