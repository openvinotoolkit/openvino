# Compressing a Model to FP16 {#openvino_docs_MO_DG_FP16_Compression}

@sphinxdirective

By default, when IR is saved all relevant floating-point weights are compressed to ``FP16`` data type during model conversion.
It results in creating a "compressed ``FP16`` model", which occupies about half of
the original space in the file system. The compression may introduce a minor drop in accuracy,
but it is negligible for most models.
In case if accuracy drop is significant user can disable compression explicitly.

To disable compression, use the ``compress_to_fp16=False`` option:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          import openvino as ov

          ov_model = ov.convert_model(original_model)
          ov.save_model(ov_model, 'model.xml' compress_to_fp16=False)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          ovc path_to_your_model --compress_to_fp16=False

For details on how plugins handle compressed ``FP16`` models, see
:doc:`Working with devices <openvino_docs_OV_UG_Working_with_devices>`.

.. note::

   ``FP16`` compression is sometimes used as the initial step for ``INT8`` quantization.
   Refer to the :doc:`Post-training optimization <**TODO: LINK TO NNCF>` guide for more
   information about that.

@endsphinxdirective


