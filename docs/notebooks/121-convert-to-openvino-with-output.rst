OpenVINO™ model conversion API
==============================

This notebook shows how to convert a model from original framework
format to OpenVINO Intermediate Representation (IR). Contents:

-  `OpenVINO IR format <#openvino-ir-format>`__
-  `IR preparation with Python conversion API and Model Optimizer
   command-line
   tool <#ir-preparation-with-python-conversion-api-and-model-optimizer-command-line-tool>`__
-  `Fetching example models <#fetching-example-models>`__
-  `Basic conversion <#basic-conversion>`__
-  `Model conversion parameters <#model-conversion-parameters>`__

   -  `Setting Input Shapes <#setting-input-shapes>`__
   -  `Cutting Off Parts of a Model <#cutting-off-parts-of-a-model>`__
   -  `Embedding Preprocessing
      Computation <#embedding-preprocessing-computation>`__

      -  `Specifying Layout <#specifying-layout>`__
      -  `Changing Model Layout <#changing-model-layout>`__
      -  `Specifying Mean and Scale
         Values <#specifying-mean-and-scale-values>`__
      -  `Reversing Input Channels <#reversing-input-channels>`__

   -  `Compressing a Model to FP16 <#compressing-a-model-to-fp>`__

-  `Convert Models Represented as Python
   Objects <#convert-models-represented-as-python-objects>`__

.. code:: ipython3

    # Required imports. Please execute this cell first.
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    "openvino-dev>=2023.1.0" "requests" "tqdm" "transformers[onnx]>=4.21.1" "torch" "torchvision"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.13.1 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3, but you have protobuf 3.20.2 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


OpenVINO IR format
------------------

OpenVINO `Intermediate Representation
(IR) <https://docs.openvino.ai/2023.0/openvino_ir.html>`__ is the
proprietary model format of OpenVINO. It is produced after converting a
model with model conversion API. Model conversion API translates the
frequently used deep learning operations to their respective similar
representation in OpenVINO and tunes them with the associated weights
and biases from the trained model. The resulting IR contains two files:
an ``.xml`` file, containing information about network topology, and a
``.bin`` file, containing the weights and biases binary data.

IR preparation with Python conversion API and Model Optimizer command-line tool
-------------------------------------------------------------------------------

There are two ways to convert a model from the original framework format
to OpenVINO IR: Python conversion API and Model Optimizer command-line
tool. You can choose one of them based on whichever is most convenient
for you. There should not be any differences in the results of model
conversion if the same set of parameters is used. For more details,
refer to `Model
Preparation <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__
documentation.

.. code:: ipython3

    # Model Optimizer CLI tool parameters description
    
    ! mo --help


.. parsed-literal::

    usage: main.py [options]
    
    optional arguments:
      -h, --help            show this help message and exit
      --framework FRAMEWORK
                            Name of the framework used to train the input model.
    
    Framework-agnostic parameters:
      --model_name MODEL_NAME, -n MODEL_NAME
                            Model_name parameter passed to the final create_ir
                            transform. This parameter is used to name a network in
                            a generated IR and output .xml/.bin files.
      --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                            Directory that stores the generated IR. By default, it
                            is the directory from where the Model Conversion is
                            launched.
      --freeze_placeholder_with_value FREEZE_PLACEHOLDER_WITH_VALUE
                            Replaces input layer with constant node with provided
                            value, for example: "node_name->True". It will be
                            DEPRECATED in future releases. Use "input" option to
                            specify a value for freezing.
      --static_shape        Enables IR generation for fixed input shape (folding
                            `ShapeOf` operations and shape-calculating sub-graphs
                            to `Constant`). Changing model input shape using the
                            OpenVINO Runtime API in runtime may fail for such an
                            IR.
      --use_new_frontend    Force the usage of new Frontend for model conversion
                            into IR. The new Frontend is C++ based and is
                            available for ONNX* and PaddlePaddle* models. Model
                            Conversion API uses new Frontend for ONNX* and
                            PaddlePaddle* by default that means `use_new_frontend`
                            and `use_legacy_frontend` options are not specified.
      --use_legacy_frontend
                            Force the usage of legacy Frontend for model
                            conversion into IR. The legacy Frontend is Python
                            based and is available for TensorFlow*, ONNX*, MXNet*,
                            Caffe*, and Kaldi* models.
      --input_model INPUT_MODEL, -w INPUT_MODEL, -m INPUT_MODEL
                            Tensorflow*: a file with a pre-trained model (binary
                            or text .pb file after freezing). Caffe*: a model
                            proto file with model weights.
      --input INPUT         Quoted list of comma-separated input nodes names with
                            shapes, data types, and values for freezing. The order
                            of inputs in converted model is the same as order of
                            specified operation names. The shape and value are
                            specified as comma-separated lists. The data type of
                            input node is specified in braces and can have one of
                            the values: f64 (float64), f32 (float32), f16
                            (float16), i64 (int64), i32 (int32), u8 (uint8),
                            boolean (bool). Data type is optional. If it's not
                            specified explicitly then there are two options: if
                            input node is a parameter, data type is taken from the
                            original node dtype, if input node is not a parameter,
                            data type is set to f32. Example, to set `input_1`
                            with shape [1,100], and Parameter node `sequence_len`
                            with scalar input with value `150`, and boolean input
                            `is_training` with `False` value use the following
                            format:
                            "input_1[1,100],sequence_len->150,is_training->False".
                            Another example, use the following format to set input
                            port 0 of the node `node_name1` with the shape [3,4]
                            as an input node and freeze output port 1 of the node
                            "node_name2" with the value [20,15] of the int32 type
                            and shape [2]:
                            "0:node_name1[3,4],node_name2:1[2]{i32}->[20,15]".
      --output OUTPUT       The name of the output operation of the model or list
                            of names. For TensorFlow*, do not add :0 to this
                            name.The order of outputs in converted model is the
                            same as order of specified operation names.
      --input_shape INPUT_SHAPE
                            Input shape(s) that should be fed to an input node(s)
                            of the model. Shape is defined as a comma-separated
                            list of integer numbers enclosed in parentheses or
                            square brackets, for example [1,3,227,227] or
                            (1,227,227,3), where the order of dimensions depends
                            on the framework input layout of the model. For
                            example, [N,C,H,W] is used for ONNX* models and
                            [N,H,W,C] for TensorFlow* models. The shape can
                            contain undefined dimensions (? or -1) and should fit
                            the dimensions defined in the input operation of the
                            graph. Boundaries of undefined dimension can be
                            specified with ellipsis, for example
                            [1,1..10,128,128]. One boundary can be undefined, for
                            example [1,..100] or [1,3,1..,1..]. If there are
                            multiple inputs in the model, --input_shape should
                            contain definition of shape for each input separated
                            by a comma, for example: [1,3,227,227],[2,4] for a
                            model with two inputs with 4D and 2D shapes.
                            Alternatively, specify shapes with the --input option.
      --example_input EXAMPLE_INPUT
                            Sample of model input in original framework. For
                            PyTorch it can be torch.Tensor. For Tensorflow it can
                            be tf.Tensor or numpy.ndarray. For PaddlePaddle it can
                            be Paddle Variable.
      --batch BATCH, -b BATCH
                            Set batch size. It applies to 1D or higher dimension
                            inputs. The default dimension index for the batch is
                            zero. Use a label 'n' in --layout or --source_layout
                            option to set the batch dimension. For example,
                            "x(hwnc)" defines the third dimension to be the batch.
      --mean_values MEAN_VALUES
                            Mean values to be used for the input image per
                            channel. Values to be provided in the (R,G,B) or
                            [R,G,B] format. Can be defined for desired input of
                            the model, for example: "--mean_values
                            data[255,255,255],info[255,255,255]". The exact
                            meaning and order of channels depend on how the
                            original model was trained.
      --scale_values SCALE_VALUES
                            Scale values to be used for the input image per
                            channel. Values are provided in the (R,G,B) or [R,G,B]
                            format. Can be defined for desired input of the model,
                            for example: "--scale_values
                            data[255,255,255],info[255,255,255]". The exact
                            meaning and order of channels depend on how the
                            original model was trained. If both --mean_values and
                            --scale_values are specified, the mean is subtracted
                            first and then scale is applied regardless of the
                            order of options in command line.
      --scale SCALE, -s SCALE
                            All input values coming from original network inputs
                            will be divided by this value. When a list of inputs
                            is overridden by the --input parameter, this scale is
                            not applied for any input that does not match with the
                            original input of the model. If both --mean_values and
                            --scale are specified, the mean is subtracted first
                            and then scale is applied regardless of the order of
                            options in command line.
      --reverse_input_channels [REVERSE_INPUT_CHANNELS]
                            Switch the input channels order from RGB to BGR (or
                            vice versa). Applied to original inputs of the model
                            if and only if a number of channels equals 3. When
                            --mean_values/--scale_values are also specified,
                            reversing of channels will be applied to user's input
                            data first, so that numbers in --mean_values and
                            --scale_values go in the order of channels used in the
                            original model. In other words, if both options are
                            specified, then the data flow in the model looks as
                            following: Parameter -> ReverseInputChannels -> Mean
                            apply-> Scale apply -> the original body of the model.
      --source_layout SOURCE_LAYOUT
                            Layout of the input or output of the model in the
                            framework. Layout can be specified in the short form,
                            e.g. nhwc, or in complex form, e.g. "[n,h,w,c]".
                            Example for many names: "in_name1([n,h,w,c]),in_name2(
                            nc),out_name1(n),out_name2(nc)". Layout can be
                            partially defined, "?" can be used to specify
                            undefined layout for one dimension, "..." can be used
                            to specify undefined layout for multiple dimensions,
                            for example "?c??", "nc...", "n...c", etc.
      --target_layout TARGET_LAYOUT
                            Same as --source_layout, but specifies target layout
                            that will be in the model after processing by
                            ModelOptimizer.
      --layout LAYOUT       Combination of --source_layout and --target_layout.
                            Can't be used with either of them. If model has one
                            input it is sufficient to specify layout of this
                            input, for example --layout nhwc. To specify layouts
                            of many tensors, names must be provided, for example:
                            --layout "name1(nchw),name2(nc)". It is possible to
                            instruct ModelOptimizer to change layout, for example:
                            --layout "name1(nhwc->nchw),name2(cn->nc)". Also "*"
                            in long layout form can be used to fuse dimensions,
                            for example "[n,c,...]->[n*c,...]".
      --compress_to_fp16 [COMPRESS_TO_FP16]
                            If the original model has FP32 weights or biases, they
                            are compressed to FP16. All intermediate data is kept
                            in original precision. Option can be specified alone
                            as "--compress_to_fp16", or explicit True/False values
                            can be set, for example: "--compress_to_fp16=False",
                            or "--compress_to_fp16=True"
      --extensions EXTENSIONS
                            Paths or a comma-separated list of paths to libraries
                            (.so or .dll) with extensions. For the legacy MO path
                            (if `--use_legacy_frontend` is used), a directory or a
                            comma-separated list of directories with extensions
                            are supported. To disable all extensions including
                            those that are placed at the default location, pass an
                            empty string.
      --transform TRANSFORM
                            Apply additional transformations. Usage: "--transform
                            transformation_name1[args],transformation_name2..."
                            where [args] is key=value pairs separated by
                            semicolon. Examples: "--transform LowLatency2" or "--
                            transform Pruning" or "--transform
                            LowLatency2[use_const_initializer=False]" or "--
                            transform "MakeStateful[param_res_names= {'input_name_
                            1':'output_name_1','input_name_2':'output_name_2'}]"
                            Available transformations: "LowLatency2",
                            "MakeStateful", "Pruning"
      --transformations_config TRANSFORMATIONS_CONFIG
                            Use the configuration file with transformations
                            description. Transformations file can be specified as
                            relative path from the current directory, as absolute
                            path or as arelative path from the mo root directory.
      --silent [SILENT]     Prevent any output messages except those that
                            correspond to log level equals ERROR, that can be set
                            with the following option: --log_level. By default,
                            log level is already ERROR.
      --log_level {CRITICAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                            Logger level of logging massages from MO. Expected one
                            of ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO',
                            'DEBUG', 'NOTSET'].
      --version             Version of Model Optimizer
      --progress [PROGRESS]
                            Enable model conversion progress display.
      --stream_output [STREAM_OUTPUT]
                            Switch model conversion progress display to a
                            multiline mode.
      --share_weights [SHARE_WEIGHTS]
                            Map memory of weights instead reading files or share
                            memory from input model. Currently, mapping feature is
                            provided only for ONNX models that do not require
                            fallback to the legacy ONNX frontend for the
                            conversion.
    
    TensorFlow*-specific parameters:
      --input_model_is_text [INPUT_MODEL_IS_TEXT]
                            TensorFlow*: treat the input model file as a text
                            protobuf format. If not specified, the Model Optimizer
                            treats it as a binary file by default.
      --input_checkpoint INPUT_CHECKPOINT
                            TensorFlow*: variables file to load.
      --input_meta_graph INPUT_META_GRAPH
                            Tensorflow*: a file with a meta-graph of the model
                            before freezing
      --saved_model_dir SAVED_MODEL_DIR
                            TensorFlow*: directory with a model in SavedModel
                            format of TensorFlow 1.x or 2.x version.
      --saved_model_tags SAVED_MODEL_TAGS
                            Group of tag(s) of the MetaGraphDef to load, in string
                            format, separated by ','. For tag-set contains
                            multiple tags, all tags must be passed in.
      --tensorflow_custom_operations_config_update TENSORFLOW_CUSTOM_OPERATIONS_CONFIG_UPDATE
                            TensorFlow*: update the configuration file with node
                            name patterns with input/output nodes information.
      --tensorflow_object_detection_api_pipeline_config TENSORFLOW_OBJECT_DETECTION_API_PIPELINE_CONFIG
                            TensorFlow*: path to the pipeline configuration file
                            used to generate model created with help of Object
                            Detection API.
      --tensorboard_logdir TENSORBOARD_LOGDIR
                            TensorFlow*: dump the input graph to a given directory
                            that should be used with TensorBoard.
      --tensorflow_custom_layer_libraries TENSORFLOW_CUSTOM_LAYER_LIBRARIES
                            TensorFlow*: comma separated list of shared libraries
                            with TensorFlow* custom operations implementation.
    
    Caffe*-specific parameters:
      --input_proto INPUT_PROTO, -d INPUT_PROTO
                            Deploy-ready prototxt file that contains a topology
                            structure and layer attributes
      --caffe_parser_path CAFFE_PARSER_PATH
                            Path to Python Caffe* parser generated from
                            caffe.proto
      --k K                 Path to CustomLayersMapping.xml to register custom
                            layers
      --disable_omitting_optional [DISABLE_OMITTING_OPTIONAL]
                            Disable omitting optional attributes to be used for
                            custom layers. Use this option if you want to transfer
                            all attributes of a custom layer to IR. Default
                            behavior is to transfer the attributes with default
                            values and the attributes defined by the user to IR.
      --enable_flattening_nested_params [ENABLE_FLATTENING_NESTED_PARAMS]
                            Enable flattening optional params to be used for
                            custom layers. Use this option if you want to transfer
                            attributes of a custom layer to IR with flattened
                            nested parameters. Default behavior is to transfer the
                            attributes without flattening nested parameters.
    
    MXNet-specific parameters:
      --input_symbol INPUT_SYMBOL
                            Symbol file (for example, model-symbol.json) that
                            contains a topology structure and layer attributes
      --nd_prefix_name ND_PREFIX_NAME
                            Prefix name for args.nd and argx.nd files.
      --pretrained_model_name PRETRAINED_MODEL_NAME
                            Name of a pretrained MXNet model without extension and
                            epoch number. This model will be merged with args.nd
                            and argx.nd files
      --save_params_from_nd [SAVE_PARAMS_FROM_ND]
                            Enable saving built parameters file from .nd files
      --legacy_mxnet_model [LEGACY_MXNET_MODEL]
                            Enable MXNet loader to make a model compatible with
                            the latest MXNet version. Use only if your model was
                            trained with MXNet version lower than 1.0.0
      --enable_ssd_gluoncv [ENABLE_SSD_GLUONCV]
                            Enable pattern matchers replacers for converting
                            gluoncv ssd topologies.
    
    Kaldi-specific parameters:
      --counts COUNTS       Path to the counts file
      --remove_output_softmax [REMOVE_OUTPUT_SOFTMAX]
                            Removes the SoftMax layer that is the output layer
      --remove_memory [REMOVE_MEMORY]
                            Removes the Memory layer and use additional inputs
                            outputs instead


.. code:: ipython3

    # Python conversion API parameters description
    from openvino.tools import mo
    
    
    mo.convert_model(help=True)


.. parsed-literal::

    Optional parameters:
      --help 
    			Print available parameters.
      --framework 
    			Name of the framework used to train the input model.
    
    Framework-agnostic parameters:
      --input_model 
    			Model object in original framework (PyTorch, Tensorflow) or path to
    			model file.
    			Tensorflow*: a file with a pre-trained model (binary or text .pb file
    			after freezing).
    			Caffe*: a model proto file with model weights
    			
    			Supported formats of input model:
    			
    			PaddlePaddle
    			paddle.hapi.model.Model
    			paddle.fluid.dygraph.layers.Layer
    			paddle.fluid.executor.Executor
    			
    			PyTorch
    			torch.nn.Module
    			torch.jit.ScriptModule
    			torch.jit.ScriptFunction
    			
    			TF
    			tf.compat.v1.Graph
    			tf.compat.v1.GraphDef
    			tf.compat.v1.wrap_function
    			tf.compat.v1.session
    			
    			TF2 / Keras
    			tf.keras.Model
    			tf.keras.layers.Layer
    			tf.function
    			tf.Module
    			tf.train.checkpoint
      --input 
    			Input can be set by passing a list of InputCutInfo objects or by a list
    			of tuples. Each tuple can contain optionally input name, input
    			type or input shape. Example: input=("op_name", PartialShape([-1,
    			3, 100, 100]), Type(np.float32)). Alternatively input can be set by
    			a string or list of strings of the following format. Quoted list of comma-separated
    			input nodes names with shapes, data types, and values for freezing.
    			If operation names are specified, the order of inputs in converted
    			model will be the same as order of specified operation names (applicable
    			for TF2, ONNX, MxNet).
    			The shape and value are specified as comma-separated lists. The data
    			type of input node is specified
    			in braces and can have one of the values: f64 (float64), f32 (float32),
    			f16 (float16), i64
    			(int64), i32 (int32), u8 (uint8), boolean (bool). Data type is optional.
    			If it's not specified explicitly then there are two options: if input
    			node is a parameter, data type is taken from the original node dtype,
    			if input node is not a parameter, data type is set to f32. Example, to set
    			`input_1` with shape [1,100], and Parameter node `sequence_len` with
    			scalar input with value `150`, and boolean input `is_training` with
    			`False` value use the following format: "input_1[1,100],sequence_len->150,is_training->False".
    			Another example, use the following format to set input port 0 of the node
    			`node_name1` with the shape [3,4] as an input node and freeze output
    			port 1 of the node `node_name2` with the value [20,15] of the int32 type
    			and shape [2]: "0:node_name1[3,4],node_name2:1[2]{i32}->[20,15]".
    			
      --output 
    			The name of the output operation of the model or list of names. For TensorFlow*,
    			do not add :0 to this name.The order of outputs in converted model is the
    			same as order of specified operation names.
      --input_shape 
    			Input shape(s) that should be fed to an input node(s) of the model. Input
    			shapes can be defined by passing a list of objects of type PartialShape,
    			Shape, [Dimension, ...] or [int, ...] or by a string of the following
    			format. Shape is defined as a comma-separated list of integer numbers
    			enclosed in parentheses or square brackets, for example [1,3,227,227]
    			or (1,227,227,3), where the order of dimensions depends on the framework
    			input layout of the model. For example, [N,C,H,W] is used for ONNX* models
    			and [N,H,W,C] for TensorFlow* models. The shape can contain undefined
    			dimensions (? or -1) and should fit the dimensions defined in the input
    			operation of the graph. Boundaries of undefined dimension can be specified
    			with ellipsis, for example [1,1..10,128,128]. One boundary can be
    			undefined, for example [1,..100] or [1,3,1..,1..]. If there are multiple
    			inputs in the model, --input_shape should contain definition of shape
    			for each input separated by a comma, for example: [1,3,227,227],[2,4]
    			for a model with two inputs with 4D and 2D shapes. Alternatively, specify
    			shapes with the --input option.
      --example_input 
    			Sample of model input in original framework.
    			For PyTorch it can be torch.Tensor.
    			For Tensorflow it can be tf.Tensor or numpy.ndarray.
    			For PaddlePaddle it can be Paddle Variable.
      --batch 
    			Set batch size. It applies to 1D or higher dimension inputs.
    			The default dimension index for the batch is zero.
    			Use a label 'n' in --layout or --source_layout option to set the batch
    			dimension.
    			For example, "x(hwnc)" defines the third dimension to be the batch.
    			
      --mean_values 
    			Mean values to be used for the input image per channel. Mean values can
    			be set by passing a dictionary, where key is input name and value is mean
    			value. For example mean_values={'data':[255,255,255],'info':[255,255,255]}.
    			Or mean values can be set by a string of the following format. Values to
    			be provided in the (R,G,B) or [R,G,B] format. Can be defined for desired
    			input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]".
    			The exact meaning and order of channels depend on how the original model
    			was trained.
      --scale_values 
    			Scale values to be used for the input image per channel. Scale values
    			can be set by passing a dictionary, where key is input name and value is
    			scale value. For example scale_values={'data':[255,255,255],'info':[255,255,255]}.
    			Or scale values can be set by a string of the following format. Values
    			are provided in the (R,G,B) or [R,G,B] format. Can be defined for desired
    			input of the model, for example: "--scale_values data[255,255,255],info[255,255,255]".
    			The exact meaning and order of channels depend on how the original model
    			was trained. If both --mean_values and --scale_values are specified,
    			the mean is subtracted first and then scale is applied regardless of
    			the order of options in command line.
      --scale 
    			All input values coming from original network inputs will be divided
    			by this value. When a list of inputs is overridden by the --input parameter,
    			this scale is not applied for any input that does not match with the original
    			input of the model. If both --mean_values and --scale  are specified,
    			the mean is subtracted first and then scale is applied regardless of
    			the order of options in command line.
      --reverse_input_channels 
    			Switch the input channels order from RGB to BGR (or vice versa). Applied
    			to original inputs of the model if and only if a number of channels equals
    			3. When --mean_values/--scale_values are also specified, reversing
    			of channels will be applied to user's input data first, so that numbers
    			in --mean_values and --scale_values go in the order of channels used
    			in the original model. In other words, if both options are specified,
    			then the data flow in the model looks as following: Parameter -> ReverseInputChannels
    			-> Mean apply-> Scale apply -> the original body of the model.
      --source_layout 
    			Layout of the input or output of the model in the framework. Layout can
    			be set by passing a dictionary, where key is input name and value is LayoutMap
    			object. Or layout can be set by string of the following format. Layout
    			can be specified in the short form, e.g. nhwc, or in complex form, e.g.
    			"[n,h,w,c]". Example for many names: "in_name1([n,h,w,c]),in_name2(nc),out_name1(n),out_name2(nc)".
    			Layout can be partially defined, "?" can be used to specify undefined
    			layout for one dimension, "..." can be used to specify undefined layout
    			for multiple dimensions, for example "?c??", "nc...", "n...c", etc.
    			
      --target_layout 
    			Same as --source_layout, but specifies target layout that will be in
    			the model after processing by ModelOptimizer.
      --layout 
    			Combination of --source_layout and --target_layout. Can't be used
    			with either of them. If model has one input it is sufficient to specify
    			layout of this input, for example --layout nhwc. To specify layouts
    			of many tensors, names must be provided, for example: --layout "name1(nchw),name2(nc)".
    			It is possible to instruct ModelOptimizer to change layout, for example:
    			--layout "name1(nhwc->nchw),name2(cn->nc)".
    			Also "*" in long layout form can be used to fuse dimensions, for example
    			"[n,c,...]->[n*c,...]".
      --compress_to_fp16 
    			If the original model has FP32 weights or biases, they are compressed
    			to FP16. All intermediate data is kept in original precision. Option
    			can be specified alone as "--compress_to_fp16", or explicit True/False
    			values can be set, for example: "--compress_to_fp16=False", or "--compress_to_fp16=True"
    			
      --extensions 
    			Paths to libraries (.so or .dll) with extensions, comma-separated
    			list of paths, objects derived from BaseExtension class or lists of
    			objects. For the legacy MO path (if `--use_legacy_frontend` is used),
    			a directory or a comma-separated list of directories with extensions
    			are supported. To disable all extensions including those that are placed
    			at the default location, pass an empty string.
      --transform 
    			Apply additional transformations. 'transform' can be set by a list
    			of tuples, where the first element is transform name and the second element
    			is transform parameters. For example: [('LowLatency2', {{'use_const_initializer':
    			False}}), ...]"--transform transformation_name1[args],transformation_name2..."
    			where [args] is key=value pairs separated by semicolon. Examples:
    			 "--transform LowLatency2" or
    			 "--transform Pruning" or
    			 "--transform LowLatency2[use_const_initializer=False]" or
    			 "--transform "MakeStateful[param_res_names=
    			{'input_name_1':'output_name_1','input_name_2':'output_name_2'}]""
    			Available transformations: "LowLatency2", "MakeStateful", "Pruning"
    			
      --transformations_config 
    			Use the configuration file with transformations description or pass
    			object derived from BaseExtension class. Transformations file can
    			be specified as relative path from the current directory, as absolute
    			path or as relative path from the mo root directory.
      --silent 
    			Prevent any output messages except those that correspond to log level
    			equals ERROR, that can be set with the following option: --log_level.
    			By default, log level is already ERROR.
      --log_level 
    			Logger level of logging massages from MO.
    			Expected one of ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO',
    			'DEBUG', 'NOTSET'].
      --version 
    			Version of Model Optimizer
      --progress 
    			Enable model conversion progress display.
      --stream_output 
    			Switch model conversion progress display to a multiline mode.
      --share_weights 
    			Map memory of weights instead reading files or share memory from input
    			model.
    			Currently, mapping feature is provided only for ONNX models
    			that do not require fallback to the legacy ONNX frontend for the conversion.
    			
    
    PaddlePaddle-specific parameters:
      --example_output 
    			Sample of model output in original framework. For PaddlePaddle it can
    			be Paddle Variable.
    
    TensorFlow*-specific parameters:
      --input_model_is_text 
    			TensorFlow*: treat the input model file as a text protobuf format. If
    			not specified, the Model Optimizer treats it as a binary file by default.
    			
      --input_checkpoint 
    			TensorFlow*: variables file to load.
      --input_meta_graph 
    			Tensorflow*: a file with a meta-graph of the model before freezing
      --saved_model_dir 
    			TensorFlow*: directory with a model in SavedModel format of TensorFlow
    			1.x or 2.x version.
      --saved_model_tags 
    			Group of tag(s) of the MetaGraphDef to load, in string format, separated
    			by ','. For tag-set contains multiple tags, all tags must be passed in.
    			
      --tensorflow_custom_operations_config_update 
    			TensorFlow*: update the configuration file with node name patterns
    			with input/output nodes information.
      --tensorflow_object_detection_api_pipeline_config 
    			TensorFlow*: path to the pipeline configuration file used to generate
    			model created with help of Object Detection API.
      --tensorboard_logdir 
    			TensorFlow*: dump the input graph to a given directory that should be
    			used with TensorBoard.
      --tensorflow_custom_layer_libraries 
    			TensorFlow*: comma separated list of shared libraries with TensorFlow*
    			custom operations implementation.
    
    MXNet-specific parameters:
      --input_symbol 
    			Symbol file (for example, model-symbol.json) that contains a topology
    			structure and layer attributes
      --nd_prefix_name 
    			Prefix name for args.nd and argx.nd files.
      --pretrained_model_name 
    			Name of a pretrained MXNet model without extension and epoch number.
    			This model will be merged with args.nd and argx.nd files
      --save_params_from_nd 
    			Enable saving built parameters file from .nd files
      --legacy_mxnet_model 
    			Enable MXNet loader to make a model compatible with the latest MXNet
    			version. Use only if your model was trained with MXNet version lower
    			than 1.0.0
      --enable_ssd_gluoncv 
    			Enable pattern matchers replacers for converting gluoncv ssd topologies.
    			
    
    Caffe*-specific parameters:
      --input_proto 
    			Deploy-ready prototxt file that contains a topology structure and
    			layer attributes
      --caffe_parser_path 
    			Path to Python Caffe* parser generated from caffe.proto
      --k 
    			Path to CustomLayersMapping.xml to register custom layers
      --disable_omitting_optional 
    			Disable omitting optional attributes to be used for custom layers.
    			Use this option if you want to transfer all attributes of a custom layer
    			to IR. Default behavior is to transfer the attributes with default values
    			and the attributes defined by the user to IR.
      --enable_flattening_nested_params 
    			Enable flattening optional params to be used for custom layers. Use
    			this option if you want to transfer attributes of a custom layer to IR
    			with flattened nested parameters. Default behavior is to transfer
    			the attributes without flattening nested parameters.
    
    Kaldi-specific parameters:
      --counts 
    			Path to the counts file
      --remove_output_softmax 
    			Removes the SoftMax layer that is the output layer
      --remove_memory 
    			Removes the Memory layer and use additional inputs outputs instead
    			
    


Fetching example models
-----------------------

This notebook uses two models for conversion examples:

-  `Distilbert <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
   NLP model from Hugging Face
-  `Resnet50 <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
   CV classification model from torchvision

.. code:: ipython3

    from pathlib import Path
    
    # create a directory for models files
    MODEL_DIRECTORY_PATH = Path("model")
    MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)

Fetch
`distilbert <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
NLP model from Hugging Face and export it in ONNX format:

.. code:: ipython3

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.onnx import export, FeaturesManager
    
    
    ONNX_NLP_MODEL_PATH = MODEL_DIRECTORY_PATH / "distilbert.onnx"
    
    # download model
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # get model onnx config function for output feature format sequence-classification
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        hf_model, feature="sequence-classification"
    )
    # fill onnx config based on pytorch model config
    onnx_config = model_onnx_config(hf_model.config)
    
    # export to onnx format
    export(
        preprocessor=tokenizer,
        model=hf_model,
        config=onnx_config,
        opset=onnx_config.default_onnx_opset,
        output=ONNX_NLP_MODEL_PATH,
    )


.. parsed-literal::

    2023-10-30 23:03:34.054449: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-10-30 23:03:34.088016: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-30 23:03:34.718197: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py:223: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mask, torch.tensor(torch.finfo(scores.dtype).min)




.. parsed-literal::

    (['input_ids', 'attention_mask'], ['logits'])



Fetch
`Resnet50 <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
CV classification model from torchvision:

.. code:: ipython3

    from torchvision.models import resnet50, ResNet50_Weights
    
    
    # create model object
    pytorch_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # switch model from training to inference mode
    pytorch_model.eval()




.. parsed-literal::

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )



Convert PyTorch model to ONNX format:

.. code:: ipython3

    import torch
    import warnings
    
    
    ONNX_CV_MODEL_PATH = MODEL_DIRECTORY_PATH / "resnet.onnx"
    
    if ONNX_CV_MODEL_PATH.exists():
        print(f"ONNX model {ONNX_CV_MODEL_PATH} already exists.")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                model=pytorch_model, args=torch.randn(1, 3, 780, 520), f=ONNX_CV_MODEL_PATH
            )
        print(f"ONNX model exported to {ONNX_CV_MODEL_PATH}")


.. parsed-literal::

    ONNX model exported to model/resnet.onnx


Basic conversion
----------------

To convert a model to OpenVINO IR, use the following command:

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/distilbert.onnx --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    # mo.convert_model returns an openvino.runtime.Model object
    ov_model = mo.convert_model(ONNX_NLP_MODEL_PATH)
    
    # then model can be serialized to *.xml & *.bin files
    from openvino.runtime import serialize
    
    serialize(ov_model, xml_path=MODEL_DIRECTORY_PATH / "distilbert.xml")


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


Model conversion parameters
---------------------------

Both Python conversion API and Model Optimizer command-line tool provide
the following capabilities: \* overriding original input shapes for
model conversion with ``input`` and ``input_shape`` parameters. `Setting
Input Shapes
guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html>`__.
\* cutting off unwanted parts of a model (such as unsupported operations
and training sub-graphs) using the ``input`` and ``output`` parameters
to define new inputs and outputs of the converted model. `Cutting Off
Parts of a Model
guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model.html>`__.
\* inserting additional input pre-processing sub-graphs into the
converted model by using the ``mean_values``, ``scales_values``,
``layout``, and other parameters. `Embedding Preprocessing Computation
article <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html>`__.
\* compressing the model weights (for example, weights for convolutions
and matrix multiplications) to FP16 data type using ``compress_to_fp16``
compression parameter. `Compression of a Model to FP16
guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html>`__.

If the out-of-the-box conversion (only the ``input_model`` parameter is
specified) is not successful, it may be required to use the parameters
mentioned above to override input shapes and cut the model.

Setting Input Shapes
~~~~~~~~~~~~~~~~~~~~

Model conversion is supported for models with dynamic input shapes that
contain undefined dimensions. However, if the shape of data is not going
to change from one inference request to another, it is recommended to
set up static shapes (when all dimensions are fully defined) for the
inputs. Doing it at this stage, instead of during inference in runtime,
can be beneficial in terms of performance and memory consumption. To set
up static shapes, model conversion API provides the ``input`` and
``input_shape`` parameters.

For more information refer to `Setting Input Shapes
guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html>`__.

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/distilbert.onnx --input input_ids,attention_mask --input_shape [1,128],[1,128] --output_dir model
    
    # alternatively
    ! mo --input_model model/distilbert.onnx --input input_ids[1,128],attention_mask[1,128] --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(
        ONNX_NLP_MODEL_PATH,
        input=["input_ids", "attention_mask"],
        input_shape=[[1, 128], [1, 128]],
    )
    
    # alternatively specify input shapes, using the input parameter
    ov_model = mo.convert_model(
        ONNX_NLP_MODEL_PATH, input=[("input_ids", [1, 128]), ("attention_mask", [1, 128])]
    )

The input_shape parameter allows overriding original input shapes to
ones compatible with a given model. Dynamic shapes, i.e. with dynamic
dimensions, can be replaced in the original model with static shapes for
the converted model, and vice versa. The dynamic dimension can be marked
in the model conversion API parameter as ``-1`` or ``?``. For example,
launch model conversion for the ONNX Bert model and specify a dynamic
sequence length dimension for inputs:

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/distilbert.onnx --input input_ids,attention_mask --input_shape [1,-1],[1,-1] --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(
        ONNX_NLP_MODEL_PATH,
        input=["input_ids", "attention_mask"],
        input_shape=[[1, -1], [1, -1]],
    )

To optimize memory consumption for models with undefined dimensions in
runtime, model conversion API provides the capability to define
boundaries of dimensions. The boundaries of undefined dimensions can be
specified with ellipsis. For example, launch model conversion for the
ONNX Bert model and specify a boundary for the sequence length
dimension:

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/distilbert.onnx --input input_ids,attention_mask --input_shape [1,10..128],[1,10..128] --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(
        ONNX_NLP_MODEL_PATH,
        input=["input_ids", "attention_mask"],
        input_shape=[[1, "10..128"], [1, "10..128"]],
    )

Cutting Off Parts of a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following examples show when model cutting is useful or even
required:

-  A model has pre- or post-processing parts that cannot be translated
   to existing OpenVINO operations.
-  A model has a training part that is convenient to be kept in the
   model but not used during inference.
-  A model is too complex to be converted at once because it contains
   many unsupported operations that cannot be easily implemented as
   custom layers.
-  A problem occurs with model conversion or inference in OpenVINO
   Runtime. To identify the issue, limit the conversion scope by an
   iterative search for problematic areas in the model.
-  A single custom layer or a combination of custom layers is isolated
   for debugging purposes.

For a more detailed description, refer to the `Cutting Off Parts of a
Model
guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model.html>`__.

.. code:: ipython3

    # Model Optimizer CLI
    
    # cut at the end
    ! mo --input_model model/distilbert.onnx --output /classifier/Gemm --output_dir model
    
    
    # cut from the beginning
    ! mo --input_model model/distilbert.onnx --input /distilbert/embeddings/LayerNorm/Add_1,attention_mask --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/distilbert.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    # cut at the end
    ov_model = mo.convert_model(ONNX_NLP_MODEL_PATH, output="/classifier/Gemm")
    
    # cut from the beginning
    ov_model = mo.convert_model(
        ONNX_NLP_MODEL_PATH,
        input=["/distilbert/embeddings/LayerNorm/Add_1", "attention_mask"],
    )

Embedding Preprocessing Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input data for inference can be different from the training dataset and
requires additional preprocessing before inference. To accelerate the
whole pipeline, including preprocessing and inference, model conversion
API provides special parameters such as ``mean_values``,
``scale_values``, ``reverse_input_channels``, and ``layout``. Based on
these parameters, model conversion API generates OpenVINO IR with
additionally inserted sub-graphs to perform the defined preprocessing.
This preprocessing block can perform mean-scale normalization of input
data, reverting data along channel dimension, and changing the data
layout. For more information on preprocessing, refer to the `Embedding
Preprocessing Computation
article <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html>`__.

Specifying Layout
^^^^^^^^^^^^^^^^^

Layout defines the meaning of dimensions in a shape and can be specified
for both inputs and outputs. Some preprocessing requires to set input
layouts, for example, setting a batch, applying mean or scales, and
reversing input channels (BGR<->RGB). For the layout syntax, check the
`Layout API
overview <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Layout_Overview.html>`__.
To specify the layout, you can use the layout option followed by the
layout value.

The following command specifies the ``NCHW`` layout for a Pytorch
Resnet50 model that was exported to the ONNX format:

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/resnet.onnx --layout nchw --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, layout="nchw")

Changing Model Layout
^^^^^^^^^^^^^^^^^^^^^

Changing the model layout may be necessary if it differs from the one
presented by input data. Use either ``layout`` or ``source_layout`` with
``target_layout`` to change the layout.

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/resnet.onnx --layout "nchw->nhwc" --output_dir model
    
    # alternatively use source_layout and target_layout parameters
    ! mo --input_model model/resnet.onnx --source_layout nchw --target_layout nhwc --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, layout="nchw->nhwc")
    
    # alternatively use source_layout and target_layout parameters
    ov_model = mo.convert_model(
        ONNX_CV_MODEL_PATH, source_layout="nchw", target_layout="nhwc"
    )

Specifying Mean and Scale Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model conversion API has the following parameters to specify the values:
``mean_values``, ``scale_values``, ``scale``. Using these parameters,
model conversion API embeds the corresponding preprocessing block for
mean-value normalization of the input data and optimizes this block so
that the preprocessing takes negligible time for inference.

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/resnet.onnx --mean_values [123,117,104] --scale 255 --output_dir model
    
    ! mo --input_model model/resnet.onnx --mean_values [123,117,104] --scale_values [255,255,255] --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, mean_values=[123, 117, 104], scale=255)
    
    ov_model = mo.convert_model(
        ONNX_CV_MODEL_PATH, mean_values=[123, 117, 104], scale_values=[255, 255, 255]
    )

Reversing Input Channels
^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, input images for your application can be of the ``RGB`` (or
``BGR``) format, and the model is trained on images of the ``BGR`` (or
``RGB``) format, which is in the opposite order of color channels. In
this case, it is important to preprocess the input images by reverting
the color channels before inference.

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/resnet.onnx --reverse_input_channels --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, reverse_input_channels=True)

Compressing a Model to FP16
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optionally all relevant floating-point weights can be compressed to FP16
data type during the model conversion, creating a compressed FP16 model.
This smaller model occupies about half of the original space in the file
system. While the compression may introduce a drop in accuracy, for most
models, this decrease is negligible.

.. code:: ipython3

    # Model Optimizer CLI
    
    ! mo --input_model model/resnet.onnx --compress_to_fp16=True --output_dir model


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/121-convert-to-openvino/model/resnet.bin


.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, compress_to_fp16=True)

Convert Models Represented as Python Objects
--------------------------------------------

Python conversion API can pass Python model objects, such as a Pytorch
model or TensorFlow Keras model directly, without saving them into files
and without leaving the training environment (Jupyter Notebook or
training scripts).

.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(pytorch_model)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


``convert_model()`` accepts all parameters available in the MO
command-line tool. Parameters can be specified by Python classes or
string analogs, similar to the command-line tool.

.. code:: ipython3

    # Python conversion API
    from openvino.tools import mo
    
    
    ov_model = mo.convert_model(
        pytorch_model,
        input_shape=[1, 3, 100, 100],
        mean_values=[127, 127, 127],
        layout="nchw",
    )
    
    ov_model = mo.convert_model(pytorch_model, source_layout="nchw", target_layout="nhwc")
    
    ov_model = mo.convert_model(
        pytorch_model, compress_to_fp16=True, reverse_input_channels=True
    )
