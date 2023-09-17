# Post-training Optimization Tool FAQ {#pot_docs_FrequentlyAskedQuestions}

@sphinxdirective

.. note:: 

   Post-training Optimization Tool has been deprecated since OpenVINO 2023.0. 
   :doc:`Neural Network Compression Framework (NNCF) <ptq_introduction>` is recommended for post-training quantization instead.


If your question is not covered below, use the `OpenVINO™ Community Forum page <https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit>`__, where you can participate freely.


.. dropdown:: Is the Post-training Optimization Tool (POT) open-sourced?

    Yes, POT is developed on GitHub as a part of `openvinotoolkit/openvino <https://github.com/openvinotoolkit/openvino>`__ under Apache-2.0 License.

.. dropdown:: Can I quantize my model without a dataset?

   In general, you should have a dataset. The dataset should be annotated if you want to validate the accuracy.
   If your dataset is not annotated, you can use :doc:`Default Quantization <pot_default_quantization_usage>` 
   to quantize the model or command-line interface with :doc:`Simplified mode <pot_docs_simplified_mode>`.

.. dropdown:: Can a model in any framework be quantized by the POT?

   The POT accepts models in the OpenVINO&trade; Intermediate Representation (IR) format only. For that you need to convert your model to the IR format using
   :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.


.. dropdown:: I'd like to quantize a model and I've converted it to IR but I don't have the Accuracy Checker config. What can I do?

   1. Try quantization using Python API of the Post-training Optimization Tool. For more details see :doc:`Default Quantization <pot_default_quantization_usage>`.
   2. If you consider command-line usage only refer to :doc:`Accuracy Checker documentation <omz_tools_accuracy_checker>` to create the Accuracy Checker configuration file, 
      and try to find the configuration file for your model among the ones available in the Accuracy Checker examples. 
   3. An alternative way is to quantize the model in the :doc:`Simplified mode <pot_docs_simplified_mode>` but you will not be able to measure the accuracy.

.. dropdown:: What is a tradeoff when you go to low precision?

   The tradeoff is between the accuracy drop and performance. When a model is in low precision, it is usually performed
   compared to the same model in full precision but the accuracy might be worse. You can find some benchmarking results in
   :doc:`INT8 vs FP32 Comparison on Select Networks and Platforms <openvino_docs_performance_int8_vs_fp32>`.
   The other benefit of having a model in low precision is its smaller size.

.. dropdown:: I tried all recommendations from "Post-Training Optimization Best Practices" but either have a high accuracy drop or bad performance after quantization. What else can I do?

   First of all, you should validate the POT compression pipeline you are running, which can be done with the following steps:

   1. Make sure the accuracy of the original uncompressed model has the value you expect. Run your POT pipeline with an empty compression config and evaluate the resulting model metric. 
      Compare this uncompressed model accuracy metric value with your reference.
   2. Run your compression pipeline with a single compression algorithm (:doc:`Default Quantization <pot_default_quantization_usage>` or :doc:`Accuracy-aware Quantization <pot_accuracyaware_usage>`) 
      without any parameter values specified in the config (except for ``preset`` and ``stat_subset_size``). Make sure you get the desirable accuracy drop/performance gain in this case.

   Finally, if you have done the steps above and the problem persists, you could try to compress your model using the 
   `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf_pytorch>`__. Note that NNCF usage requires you to have a 
   PyTorch or TensorFlow 2 based training pipeline of your model to perform Quantization-aware Training. 
   See :doc:`Model Optimization Guide <openvino_docs_model_optimization_guide>` for more details.

.. dropdown:: I get “RuntimeError: Cannot get memory” and “RuntimeError: Output data was not allocated” when I quantize my model by the POT.

   These issues happen due to insufficient available amount of memory for statistics collection during the quantization process of a huge model or
   due to a very high resolution of input images in the quantization dataset. If you do not have a possibility to increase your RAM size, one of the following options can help:

   - Set ``inplace_statistics`` parameters to ``True``. In that case, the POT will change the method to collect statistics and use less memory. 
     Note that such change might increase the time required for quantization.
   - Set ``eval_requests_number`` and ``stat_requests_number`` parameters to 1. In that case, the POT will limit the number of infer requests by 1 and use less memory.
     Note that such change might increase the time required for quantization.
   - Set ``use_fast_bias`` parameter to ``false``. In that case, the POT will switch from the FastBiasCorrection algorithm to the full BiasCorrection algorithm
     which is usually more accurate and takes more time but requires less memory. See :doc:`Post-Training Optimization Best Practices <pot_docs_BestPractices>` for more details.
   - Reshape your model to a lower resolution and resize the size of images in the dataset. Note that such change might impact the accuracy.

.. dropdown:: I have successfully quantized my model with a low accuracy drop and improved performance but the output video generated from the low precision model is much worse than from the full precision model. What could be the root cause?

   It can happen due to the following reasons:
   
   - A wrong or not representative dataset was used during the quantization and accuracy validation. 
     Please make sure that your data and labels are correct and they sufficiently reflect the use case.
   - If the command-line interface was used for quantization, a wrong Accuracy Checker configuration file could lead to this problem. 
     Refer to :doc:`Accuracy Checker documentation <omz_tools_accuracy_checker>` for more information.
   - If :doc:`Default Quantization <pot_default_quantization_usage>` was used for quantization you can also try 
     :doc:`Accuracy-aware Quantization <pot_accuracyaware_usage>` method that allows controlling maximum accuracy deviation.

.. dropdown:: The quantization process of my model takes a lot of time. Can it be decreased somehow?

   Quantization time depends on multiple factors such as the size of the model and the dataset. It also depends on the algorithm:
   the :doc:`Default Quantization <pot_default_quantization_usage>` algorithm takes less time than the :doc:`Accuracy-aware Quantization <pot_accuracyaware_usage>` algorithm.
   The following configuration parameters also impact the quantization time duration
   (see details in :doc:`Post-Training Optimization Best Practices <pot_docs_BestPractices>`):
   
   - ``use_fast_bias``: when set to ``false``, it increases the quantization time
   - ``stat_subset_size``: the higher the value of this parameter, the more time will be required for the quantization
   - ``tune_hyperparams``: if set to ``true`` when the AccuracyAwareQuantization algorithm is used, it increases the quantization time
   - ``stat_requests_number``: the lower number, the more time might be required for the quantization
   - ``eval_requests_number``: the lower number, the more time might be required for the quantization

   Note that higher values of ``stat_requests_number`` and ``eval_requests_number`` increase memory consumption by POT.

.. dropdown:: When I execute POT CLI, I get "File "/workspace/venv/lib/python3.7/site-packages/nevergrad/optimization/base.py", line 35... SyntaxError: invalid syntax". What is wrong?

   This error is reported when you have a Python version older than 3.7 in your environment. Upgrade your Python version.

.. dropdown:: What does the message "ModuleNotFoundError: No module named 'some\_module\_name'" mean?

   It means that some required python module is not installed in your environment. To install it, run ``pip install some_module_name``.

.. dropdown:: Is there a way to collect an intermediate IR when the AccuracyAware mechanism fails?

   You can add ``"dump_intermediate_model": true`` to the POT configuration file and it will drop an intermediate IR to ``accuracy_aware_intermediate`` folder.

.. dropdown:: What do the messages "Output name: result_operation_name not found" or "Output node with result_operation_name is not found in graph" mean?

   Errors are caused by missing output nodes names in a graph when using the POT tool for model quantization. 
   It might appear for some models only for IRs converted from ONNX models using the new frontend (which is the default 
   conversion path starting from 2022.1 release). To avoid such errors, use the legacy MO frontend to convert a model 
   to IR by passing the ``--use_legacy_frontend`` option. Then, use the produced IR for quantization.

@endsphinxdirective

