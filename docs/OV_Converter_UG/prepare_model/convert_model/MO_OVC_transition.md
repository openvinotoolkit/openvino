# Transition from Legacy Conversion API {#openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition}

@sphinxdirective

.. meta::
   :description: Transition guide from MO / mo.convert_model() to OVC / ov.convert_model().

In 2023.1 OpenVino release a new OVC (OpenVINO Model Converter) tool was introduced with the corresponding Python API: ``openvino.convert_model`` method. ``ovc`` and ``openvino.convert_model`` represent
a lightweight alternative of ``mo`` and ``openvino.tools.mo.convert_model`` which are considered legacy API now. In this article, all the differences between ``mo`` and ``ovc`` are summarized and help in the transition from the legacy API to the new API is provided.

MO vs OVC parameters comparison
###############################

The comparison of parameters between ov.convert_model() / OVC and mo.convert_model() / MO.

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Parameter name in mo.convert_model() / MO
     - Parameter name in ov.convert_model() / OVC
     - Differences description
   * - input_model
     - input_model
     - Along with model object or path to input model ov.convert_model() accepts list of model parts, for example path to Tensorflow weights plus path to Tensorflow checkpoint. OVC tool accepts unnamed input model.
   * - output_dir
     - output_model
     - output_model in OVC tool sets both output model name and output directory.
   * - model_name
     - output_model
     - output_model in OVC tool sets both output model name and output directory.
   * - input
     - input
     - ov.convert_model() accepts tuples for setting multiple parameters. OVC tool 'input' does not have type setting and freezing functionality. ov.convert_model() does not allow input cut.
   * - output
     - output
     - ov.convert_model() does not allow output cut.
   * - input_shape
     - -
     - -
   * - example_input
     - example_input
     - -
   * - batch
     - -
     - -
   * - mean_values
     - -
     - -
   * - scale_values
     - -
     - -
   * - scale
     - -
     - -
   * - reverse_input_channels
     - -
     - -
   * - source_layout
     - -
     - -
   * - target_layout
     - -
     - -
   * - layout
     - -
     - -
   * - compress_to_fp16
     - compress_to_fp16
     - OVC provides 'compress_to_fp16' for command line tool only, as compression is performed during saving a model to IR (Intermediate Representation).
   * - extensions
     - extension
     - -
   * - transform
     - -
     - -
   * - transformations_config
     - -
     - -
   * - static_shape
     - -
     - -
   * - freeze_placeholder_with_value
     - -
     - -
   * - use_legacy_frontend
     - -
     - -
   * - use_legacy_frontend
     - -
     - -
   * - silent
     - verbose
     - OVC / ov.convert_model provides 'verbose' parameter instead of 'silent' for printing of detailed conversion information if 'verbose' is set to True.
   * - log_level
     - -
     - -
   * - version
     - version
     - -
   * - progress
     - -
     - -
   * - stream_output
     - -
     - -
   * - share_weights
     - share_weights
     - -
   * - framework
     - -
     - -
   * - help / -h
     - help / -h
     - OVC provides help parameter only in command line tool.
   * - example_output
     - output
     - OVC / ov.convert_model 'output' parameter includes capabilities of MO 'example_output' parameter.
   * - input_model_is_text
     - -
     - -
   * - input_checkpoint
     - input_model
     - All supported model formats can be passed to 'input_model'.
   * - input_meta_graph
     - input_model
     - All supported model formats can be passed to 'input_model'.
   * - saved_model_dir
     - input_model
     - All supported model formats can be passed to 'input_model'.
   * - saved_model_tags
     - -
     - -
   * - tensorflow_custom_operations_config_update
     - -
     - -
   * - tensorflow_object_detection_api_pipeline_config
     - -
     - -
   * - tensorboard_logdir
     - -
     - -
   * - tensorflow_custom_layer_libraries
     - -
     - -
   * - input_symbol
     - -
     - -
   * - nd_prefix_name
     - -
     - -
   * - pretrained_model_name
     - -
     - -
   * - save_params_from_nd
     - -
     - -
   * - legacy_mxnet_model
     - -
     - -
   * - enable_ssd_gluoncv
     - -
     - -
   * - input_proto
     - -
     - -
   * - caffe_parser_path
     - -
     - -
   * - k
     - -
     - -
   * - disable_omitting_optional
     - -
     - -
   * - enable_flattening_nested_params
     - -
     - -
   * - counts
     - -
     - -
   * - remove_output_softmax
     - -
     - -
   * - remove_memory
     - -
     - -

Preprocessing of model using mo.convert_model() vs using ov.convert_model().
############################################################################

mo.convert_model() provides a wide range of preprocessing parameters. Most of these parameters have analogs in ``ov.PrePostProcessor`` class.
Here is the list of MO parameters which can be replaced with usage of ``ov.PrePostProcessor`` class.

* mean_value parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).tensor().set_layout(ov.Layout(layout_value))
             prep.input(input_name).preprocess().mean([0.5, 0.5, 0.5])
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools import mo
             ov_model = mo.convert_model(model, mean_values=[0.5, 0.5, 0.5])

* scale_value parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).tensor().set_layout(ov.Layout(layout_value))
             prep.input(input_name).preprocess().scale([255., 255., 255.])
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools import mo
             ov_model = mo.convert_model(model, scale_value=[255., 255., 255.])

* scale parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).preprocess().scale(255.)
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools import mo
             ov_model = mo.convert_model(model, scale=255.)

* reverse_input_channels parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).tensor().set_layout(ov.Layout(layout_value))
             prep.input(input_name).preprocess().reverse_channels()
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools import mo
             ov_model = mo.convert_model(model, reverse_input_channels=True)

* source_layout parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).model().set_layout(ov.Layout("nhwc"))
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             from openvino.tools import mo
             
             ov_model = mo.convert_model(model, source_layout={input_name: ov.Layout("nhwc")})

* target_layout parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).tensor().set_layout(ov.Layout("nhwc"))
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             from openvino.tools import mo
             
             ov_model = mo.convert_model(model, target_layout={input_name: ov.Layout("nhwc")})


* layout parameter:

   .. tab-set::

       .. tab-item:: ov.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             import openvino as ov
             ov_model = ov.convert_model(model)

             prep = ov.preprocess.PrePostProcessor(ov_model)
             prep.input(input_name).model().set_layout(ov.Layout("nchw"))
             prep.input(input_name).tensor().set_layout(ov.Layout("nhwc"))
             ov_model = prep.build()

       .. tab-item:: mo.convert_model()
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools import mo
             ov_model = mo.convert_model(model, layout={input_name: mo.LayoutMap("nchw", "nhwc")})

MO vs OVC model formats
#######################

ov.convert_model() and OVC tool support conversion from PyTorch, TF, TF Lite, ONNX, PaddlePaddle.
Following frameworks are supported only in MO and mo.convert_model(): Caffe, MxNet, Kaldi, which are deprecated.

@endsphinxdirective


