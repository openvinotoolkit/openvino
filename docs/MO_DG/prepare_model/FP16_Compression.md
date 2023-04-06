# Compressing a Model to FP16 {#openvino_docs_MO_DG_FP16_Compression}

@sphinxdirective

Model Optimizer by default converts all floating-point weights to ``FP16`` data type. 
The resulting IR is called compressed ``FP16`` model. The resulting model will occupy 
about twice as less space in the file system, but it may have some accuracy drop. 
For most models, the accuracy drop is negligible. But in case if accuracy drop is 
significant user can disable compression explicitly.

By default, models are compressed to ``FP16``, but you can disable compression by 
specifying ``--compress_to_fp16=False``:

.. code-block:: sh

   mo --input_model INPUT_MODEL --compress_to_fp16=False


For details on how plugins handle compressed ``FP16`` models, see 
:doc:`Working with devices <openvino_docs_OV_UG_Working_with_devices>`.

.. note::

   ``FP16`` compression is sometimes used as the initial step for ``INT8`` quantization. 
   Refer to the :doc:`Post-training optimization <pot_introduction>` guide for more 
   information about that.


@endsphinxdirective
