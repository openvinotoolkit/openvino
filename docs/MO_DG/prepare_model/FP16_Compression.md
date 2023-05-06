# Compressing a Model to FP16 {#openvino_docs_MO_DG_FP16_Compression}

@sphinxdirective

Model Optimizer can convert all floating-point weights to the ``FP16`` data type. 
It results in creating a "compressed ``FP16`` model", which occupies about half of 
the original space in the file system. The compression may introduce a drop in accuracy.
but it is negligible for most models.

To compress the model, use the `--compress_to_fp16` or `--compress_to_fp16=True` option:

.. code-block:: sh

   mo --input_model INPUT_MODEL --compress_to_fp16


For details on how plugins handle compressed ``FP16`` models, see 
:doc:`Working with devices <openvino_docs_OV_UG_Working_with_devices>`.

.. note::

   ``FP16`` compression is sometimes used as the initial step for ``INT8`` quantization. 
   Refer to the :doc:`Post-training optimization <pot_introduction>` guide for more 
   information about that.


.. note::

   Some large models (larger than a few GB) when compressed to ``FP16`` may consume enormous amount of RAM on the loading
   phase of the inference. In case if you are facing such problems, please try to convert them without compression: 
   `mo --input_model INPUT_MODEL --compress_to_fp16=False`


@endsphinxdirective
