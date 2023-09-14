Quantization of Image Classification Models
===========================================

This tutorial demonstrates how to apply ``INT8`` quantization to Image
Classification model using
`NNCF <https://github.com/openvinotoolkit/nncf>`__. It uses the
MobileNet V2 model, trained on Cifar10 dataset. The code is designed to
be extendable to custom models and datasets. The tutorial uses OpenVINO
backend for performing model quantization in NNCF, if you interested how
to apply quantization on PyTorch model, please check this
`tutorial <112-pytorch-post-training-quantization-nncf-with-output.html>`__.

This tutorial consists of the following steps:

- Prepare the model for quantization.
- Define a data loading functionality.
- Perform quantization.
- Compare accuracy of the original and quantized models.
- Compare performance of the original and quantized models.
- Compare results on one picture.

**Table of contents:**

- `Prepare the Model <#prepare-the-model>`__
- `Prepare Dataset <#prepare-dataset>`__
- `Perform Quantization <#perform-quantization>`__

  - `Create Dataset for Validation <#create-dataset-for-validation>`__

- `Run nncf.quantize for Getting an Optimized Model <#run-nncf.quantize-for-getting-an-optimized-model>`__
- `Serialize an OpenVINO IR model <#serialize-an-openvino-ir-model>`__
- `Compare Accuracy of the Original and Quantized Models <#compare-accuracy-of-the-original-and-quantized-models>`__

  - `Select inference device <#select-inference-device>`__

- `Compare Performance of the Original and Quantized Models <#compare-performance-of-the-original-and-quantized-models>`__
- `Compare results on four pictures <#compare-results-on-four-pictures>`__

.. code:: ipython3

    # Install openvino package
    !pip install -q "openvino==2023.1.0.dev20230811"

.. code:: ipython3

    from pathlib import Path
    
    # Set the data and model directories
    DATA_DIR = Path('../data/datasets/cifar10')
    MODEL_DIR = Path('model')
    model_repo = 'pytorch-cifar-models'
    
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

Prepare the Model
###############################################################################################################################

Model preparation stage has the following steps:

-  Download a PyTorch model
-  Convert model to OpenVINO Intermediate Representation format (IR)
   using model conversion Python API
-  Serialize converted model on disk

.. code:: ipython3

    import sys
    
    if not Path(model_repo).exists():
        !git clone https://github.com/chenyaofo/pytorch-cifar-models.git
    
    sys.path.append(model_repo)


.. parsed-literal::

    Cloning into 'pytorch-cifar-models'...
    remote: Enumerating objects: 282, done.[K
    remote: Counting objects: 100% (281/281), done.[K
    remote: Compressing objects: 100% (96/96), done.[K
    remote: Total 282 (delta 135), reused 269 (delta 128), pack-reused 1[K
    Receiving objects: 100% (282/282), 9.22 MiB | 3.92 MiB/s, done.
    Resolving deltas: 100% (135/135), done.


.. code:: ipython3

    from pytorch_cifar_models import cifar10_mobilenetv2_x1_0
    
    model = cifar10_mobilenetv2_x1_0(pretrained=True)

OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation format using model conversion Python API.
``ov.convert_model`` accept PyTorch model instance and convert it into
``openvino.runtime.Model`` representation of model in OpenVINO.
Optionally, you may specify ``example_input`` which serves as a helper
for model tracing and ``input_shape`` for converting the model with
static shape. The converted model is ready to be loaded on a device for
inference and can be saved on a disk for next usage via the
``save_model`` function. More details about model conversion Python API
can be found on this
`page <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.

.. code:: ipython3

    import openvino as ov
    
    model.eval()
    
    ov_model = ov.convert_model(model, input=[1,3,32,32])
    
    ov.save_model(ov_model, MODEL_DIR / "mobilenet_v2.xml") 


.. parsed-literal::

    2023-09-08 23:00:34.215999: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-08 23:00:34.251815: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-08 23:00:34.795978: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


Prepare Dataset
###############################################################################################################################

We will use `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__
dataset from
`torchvision <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>`__.
Preprocessing for model obtained from training
`config <https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf>`__

.. code:: ipython3

    import torch
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    dataset = CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


.. parsed-literal::

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data/datasets/cifar10/cifar-10-python.tar.gz



.. parsed-literal::

      0%|          | 0/170498071 [00:00<?, ?it/s]


.. parsed-literal::

    Extracting ../data/datasets/cifar10/cifar-10-python.tar.gz to ../data/datasets/cifar10


Perform Quantization
###############################################################################################################################

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
MobileNetV2. The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the ``openvino.save_model``
   function.

Create Dataset for Validation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

NNCF is compatible with ``torch.utils.data.DataLoader`` interface. For
performing quantization it should be passed into ``nncf.Dataset`` object
with transformation function, which prepares input data to fit into
model during quantization, in our case, to pick input tensor from pair
(input tensor and label) and convert PyTorch tensor to numpy.

.. code:: ipython3

    import nncf
    
    def transform_fn(data_item):
        image_tensor = data_item[0]
        return image_tensor.numpy()
    
    quantization_dataset = nncf.Dataset(val_loader, transform_fn)

Run nncf.quantize for Getting an Optimized Model
###############################################################################################################################

``nncf.quantize`` function accepts model and prepared quantization
dataset for performing basic quantization. Optionally, additional
parameters like ``subset_size``, ``preset``, ``ignored_scope`` can be
provided to improve quantization result if applicable. More details
about supported parameters can be found on this
`page <https://docs.openvino.ai/2023.0/basic_quantization_flow.html#tune-quantization-parameters>`__

.. code:: ipython3

    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:08<00:00, 35.19it/s]
    Biases correction: 100%|██████████| 36/36 [00:01<00:00, 21.91it/s]


Serialize an OpenVINO IR model
###############################################################################################################################

Similar to ``ov.convert_model``, quantized model is ``ov.Model`` object
which ready to be loaded into device and can be serialized on disk using
``ov.save_model``.

.. code:: ipython3

    ov.save_model(quant_ov_model, MODEL_DIR / "quantized_mobilenet_v2.xml")

Compare Accuracy of the Original and Quantized Models
###############################################################################################################################

.. code:: ipython3

    from tqdm.notebook import tqdm
    import numpy as np
    
    def test_accuracy(ov_model, data_loader):
        correct = 0
        total = 0
        for (batch_imgs, batch_labels) in tqdm(data_loader):
            result = ov_model(batch_imgs)[0]
            top_label = np.argmax(result)
            correct += top_label == batch_labels.numpy()
            total += 1
        return correct / total

Select inference device
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    core = ov.Core()
    compiled_model = core.compile_model(ov_model, device.value)
    optimized_compiled_model = core.compile_model(quant_ov_model, device.value)
    
    orig_accuracy = test_accuracy(compiled_model, val_loader)
    optimized_accuracy = test_accuracy(optimized_compiled_model, val_loader)



.. parsed-literal::

      0%|          | 0/10000 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/10000 [00:00<?, ?it/s]


.. code:: ipython3

    print(f"Accuracy of the original model: {orig_accuracy[0] * 100 :.2f}%")
    print(f"Accuracy of the optimized model: {optimized_accuracy[0] * 100 :.2f}%")


.. parsed-literal::

    Accuracy of the original model: 93.61%
    Accuracy of the optimized model: 93.54%


Compare Performance of the Original and Quantized Models
###############################################################################################################################

Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
- an inference performance measurement tool in OpenVINO.

.. note::
    
   For more accurate performance, it is recommended to run
   benchmark_app in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change CPU to GPU to benchmark
   on GPU. Run ``benchmark_app --help`` to see an overview of all
   command-line options.

.. code:: ipython3

    # Inference FP16 model (OpenVINO IR)
    !benchmark_app -m "model/mobilenet_v2.xml" -d $device.value -api async -t 15


.. parsed-literal::

    /bin/bash: benchmark_app: command not found


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    !benchmark_app -m "model/quantized_mobilenet_v2.xml" -d $device.value -api async -t 15


.. parsed-literal::

    /bin/bash: benchmark_app: command not found


Compare results on four pictures
###############################################################################################################################

.. code:: ipython3

    # Define all possible labels from the CIFAR10 dataset
    labels_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    all_pictures = []
    all_labels = []
    
    # Get all pictures and their labels.
    for i, batch in enumerate(val_loader):
        all_pictures.append(batch[0].numpy())
        all_labels.append(batch[1].item())

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    def plot_pictures(indexes: list, all_pictures=all_pictures, all_labels=all_labels):
        """Plot 4 pictures.
        :param indexes: a list of indexes of pictures to be displayed.
        :param all_batches: batches with pictures.
        """
        images, labels = [], []
        num_pics = len(indexes)
        assert num_pics == 4, f'No enough indexes for pictures to be displayed, got {num_pics}'
        for idx in indexes:
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            pic = np.rollaxis(all_pictures[idx].squeeze(), 0, 3)
            images.append(pic)
    
            labels.append(labels_names[all_labels[idx]])
    
        f, axarr = plt.subplots(1, 4)
        axarr[0].imshow(images[0])
        axarr[0].set_title(labels[0])
    
        axarr[1].imshow(images[1])
        axarr[1].set_title(labels[1])
    
        axarr[2].imshow(images[2])
        axarr[2].set_title(labels[2])
    
        axarr[3].imshow(images[3])
        axarr[3].set_title(labels[3])

.. code:: ipython3

    def infer_on_pictures(model, indexes: list, all_pictures=all_pictures):
        """ Inference model on a few pictures.
        :param net: model on which do inference
        :param indexes: list of indexes 
        """
        output_key = model.output(0)
        predicted_labels = []
        for idx in indexes:
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            result = model(all_pictures[idx])[output_key]
            result = labels_names[np.argmax(result[0])]
            predicted_labels.append(result)
        return predicted_labels

.. code:: ipython3

    indexes_to_infer = [7, 12, 15, 20]  # To plot, specify 4 indexes.
    
    plot_pictures(indexes_to_infer)
    
    results_float = infer_on_pictures(compiled_model, indexes_to_infer)
    results_quanized = infer_on_pictures(optimized_compiled_model, indexes_to_infer)
    
    print(f"Labels for picture from float model : {results_float}.")
    print(f"Labels for picture from quantized model : {results_quanized}.")


.. parsed-literal::

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


.. parsed-literal::

    Labels for picture from float model : ['frog', 'dog', 'ship', 'horse'].
    Labels for picture from quantized model : ['frog', 'dog', 'ship', 'horse'].



.. image:: 113-image-classification-quantization-with-output_files/113-image-classification-quantization-with-output_30_2.png

