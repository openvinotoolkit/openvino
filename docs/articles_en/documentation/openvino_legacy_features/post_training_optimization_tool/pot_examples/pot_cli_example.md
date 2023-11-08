# End-to-end Command-line Interface Example {#pot_configs_examples_README}

@sphinxdirective

This tutorial describes an example of running post-training quantization for the **MobileNet v2 model from PyTorch** framework, 
particularly by the DefaultQuantization algorithm.
The example covers the following steps:

- Environment setup
- Model preparation and converting it to the OpenVINO™ Intermediate Representation (IR) format
- Performance benchmarking of the original full-precision model
- Dataset preparation
- Accuracy validation of the full-precision model in the IR format
- Model quantization by the DefaultQuantization algorithm and accuracy validation of the quantized model
- Performance benchmarking of the quantized model

All the steps are based on the tools and samples of configuration files distributed with the Intel® Distribution of OpenVINO™ toolkit.

The example has been verified in Ubuntu 18.04 Operating System with Python 3.6 installed.

In case of issues while running the example, refer to :doc:`POT Frequently Asked Questions <pot_docs_FrequentlyAskedQuestions>` for help.

Model Preparation
####################

1. Navigate to ``<EXAMPLE_DIR>``.

2. Download the MobileNet v2 PyTorch model using :doc:`Model Downloader <omz_tools_downloader>` tool from the Open Model Zoo repository:

   .. code-block:: sh

      omz_downloader --name mobilenet-v2-pytorch


   After that, the original full-precision model is located in ``<EXAMPLE_DIR>/public/mobilenet-v2-pytorch/``.

3. Convert the model to the OpenVINO™ Intermediate Representation (IR) format using :doc:`Model Converter <omz_tools_downloader>` tool:

   .. code-block:: sh

      omz_converter --name mobilenet-v2-pytorch


   After that, the full-precision model in the IR format is located in ``<EXAMPLE_DIR>/public/mobilenet-v2-pytorch/FP32/``.

For more information about Model Conversion API, refer to its :doc:`documentation <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.

Performance Benchmarking of Full-Precision Models
#################################################

Check the performance of the full-precision model in the IR format using :doc:`Deep Learning Benchmark <openvino_inference_engine_tools_benchmark_tool_README>` tool:

.. code-block:: sh

   benchmark_app -m <EXAMPLE_DIR>/public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml

Note that the results might be different depending on the characteristics of your machine. On a machine with Intel® Core™ i9-10920X CPU @ 3.50GHz it is like:

.. code-block:: sh

   Latency:    4.14 ms
   Throughput: 1436.55 FPS


Dataset Preparation
####################

To perform the accuracy validation as well as quantization of a model, the dataset should be prepared. This example uses a real dataset called ImageNet. 

To download images:

1. Go to the `ImageNet <http://www.image-net.org/>`__ homepage.
2. If you do not have an account, click the ``Signup`` button in the right upper corner, provide your data, and wait for a confirmation email.
3. Log in after receiving the confirmation email or if you already have an account. Go to the ``Download`` tab.
4. Select ``Download Original Images``.
5. You will be redirected to the ``Terms of Access`` page. If you agree to the Terms, continue by clicking ``Agree and Sign``.
6. Click one of the links in the ``Download as one tar file`` section.
7. Unpack the downloaded archive into ``<EXAMPLE_DIR>/ImageNet/``.

Note that the registration process might be quite long.

Note that the ImageNet size is 50 000 images and takes around 6.5 GB of disk space.

To download the annotation file:

1. Download `archive <http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz>`__.
2. Unpack ``val.txt`` from the archive into ``<EXAMPLE_DIR>/ImageNet/``.

After that, the ``<EXAMPLE_DIR>/ImageNet/`` dataset folder should have a lot of image files like ``ILSVRC2012_val_00000001.JPEG`` and the ``val.txt`` annotation file.

Accuracy Validation of Full-Precision Model in IR Format
########################################################

1. Create a new file in ``<EXAMPLE_DIR>`` and name it ``mobilenet_v2_pytorch.yaml``. This is the Accuracy Checker configuration file.

2. Put the following text into ``mobilenet_v2_pytorch.yaml`` :

   .. code-block:: sh

      models:
        - name: mobilenet-v2-pytorch

          launchers:
            - framework: dlsdk
              device: CPU
              adapter: classification

          datasets:
            - name: classification_dataset
              data_source: ./ImageNet
              annotation_conversion:
                converter: imagenet
                annotation_file: ./ImageNet/val.txt
              reader: pillow_imread

              preprocessing:
                - type: resize
                  size: 256
                  aspect_ratio_scale: greater
                  use_pillow: True
                - type: crop
                  size: 224
                  use_pillow: True
                - type: bgr_to_rgb

              metrics:
                - name: accuracy@top1
                  type: accuracy
                  top_k: 1

                - name: accuracy@top5
                  type: accuracy
                  top_k: 5


   where ``data_source: ./ImageNet`` is the dataset and ``annotation_file: ./ImageNet/val.txt`` 
   is the annotation file prepared in the previous step. For more information about 
   the Accuracy Checker configuration file refer to :doc:`Accuracy Checker Tool documentation <omz_tools_accuracy_checker>`.

3. Evaluate the accuracy of the full-precision model in the IR format by executing the following command in ``<EXAMPLE_DIR>`` :

   .. code-block:: sh

      accuracy_check -c mobilenet_v2_pytorch.yaml -m ./public/mobilenet-v2-pytorch/FP32/


   The actual result should be like **71.81%** of the accuracy top-1 metric on VNNI-based CPU.
   Note that the results might be different on CPUs with different instruction sets.


Model Quantization
####################

1. Create a new file in ``<EXAMPLE_DIR>`` and name it ``mobilenet_v2_pytorch_int8.json``. This is the POT configuration file.

2. Put the following text into ``mobilenet_v2_pytorch_int8.json`` :

   .. code-block:: sh

      {
          "model": {
              "model_name": "mobilenet-v2-pytorch",
              "model": "./public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml",
              "weights": "./public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.bin"
          },
          "engine": {
              "config": "./mobilenet_v2_pytorch.yaml"
          },
          "compression": {
              "algorithms": [
                  {
                      "name": "DefaultQuantization",
                      "params": {
                          "preset": "mixed",
                          "stat_subset_size": 300
                      }
                  }
              ]
          }
      }


   where ``"model": "./public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml"`` and 
   ``"weights": "./public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.bin"`` specify 
   the full-precision model in the IR format, ``"config": "./mobilenet_v2_pytorch.yaml"`` 
   is the Accuracy Checker configuration file, and  ``"name": "DefaultQuantization"`` is the algorithm name.

3. Perform model quantization by executing the following command in ``<EXAMPLE_DIR>`` :

   .. code-block:: sh

      pot -c mobilenet_v2_pytorch_int8.json -e


   The quantized model is placed into the subfolder with your current date and time 
   in the name under the ``./results/mobilenetv2_DefaultQuantization/`` directory.
   The accuracy validation of the quantized model is performed right after the quantization. 
   The actual result should be like **71.556%** of the accuracy top-1 metric on VNNI-based CPU.
   Note that the results might be different on CPUs with different instruction sets.


Performance Benchmarking of Quantized Model
###########################################

Check the performance of the quantized model using :doc:`Deep Learning Benchmark <openvino_inference_engine_tools_benchmark_tool_README>` tool:

.. code-block:: sh

   benchmark_app -m <INT8_MODEL>


where ``<INT8_MODEL>`` is the path to the quantized model.
Note that the results might be different depending on the characteristics of your 
machine. On a machine with Intel® Core™ i9-10920X CPU @ 3.50GHz it is like:

.. code-block:: sh

   Latency:    1.54 ms
   Throughput: 3814.18 FPS


@endsphinxdirective
