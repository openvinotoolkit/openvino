# Compressing a Model to FP16 {#openvino_docs_MO_DG_FP16_Compression}

@sphinxdirective

Optionally, all relevant floating-point weights can be compressed to ``FP16`` data type during model conversion.
It results in creating a "compressed ``FP16`` model", which occupies about half of 
the original space in the file system. The compression may introduce a minor drop in accuracy,
but it is negligible for most models.

To compress the model, use the ``compress_to_fp16=True`` option:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model(INPUT_MODEL, compress_to_fp16=True)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model INPUT_MODEL --compress_to_fp16=True


For details on how plugins handle compressed ``FP16`` models, see 
:doc:`Working with devices <openvino_docs_OV_UG_Working_with_devices>`.

.. note::

   ``FP16`` compression is sometimes used as the initial step for ``INT8`` quantization. 
   Refer to the :doc:`Post-training optimization <pot_introduction>` guide for more 
   information about that.


.. note::

   Some large models (larger than a few GB) when compressed to ``FP16`` may consume an overly large amount of RAM on the loading
   phase of the inference. If that is the case for your model, try to convert it without compression: 
   ``convert_model(INPUT_MODEL, compress_to_fp16=False)`` or ``convert_model(INPUT_MODEL)``


@endsphinxdirective
