.. {#openvino_docs_MO_DG_prepare_model_convert_model_mxnet_specific_Convert_Style_Transfer_From_MXNet}

Converting an MXNet Style Transfer Model
========================================


.. meta::
   :description: Learn how to convert a Style Transfer 
                 model from MXNet to the OpenVINO Intermediate Representation.


.. warning::

   Note that OpenVINO support for Apache MXNet is currently being deprecated and will be removed entirely in the future.

This article provides instructions on how to generate a model for style transfer, using the public MXNet neural style transfer sample.

**Step 1**: Download or clone the repository `Zhaw's Neural Style Transfer repository <https://github.com/zhaw/neural_style>`__ with an MXNet neural style transfer sample.

**Step 2**: Prepare the environment required to work with the cloned repository:

.. note::

   Python-tk installation is needed only for Linux. Python for Windows includes it by default.


1. Install packages dependency.

   .. code-block:: sh

      sudo apt-get install python-tk


2. Install Python requirements:

   .. code-block:: sh

      pip3 install --user mxnet
      pip3 install --user matplotlib
      pip3 install --user scikit-image


**Step 3**: Download the pre-trained `VGG19 model <https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params>`__ and save it to the root directory of the cloned repository. The sample expects the model ``vgg19.params`` file to be in that directory.

**Step 4**: Modify source code files of style transfer sample from the cloned repository:

1. Go to the ``fast_mrf_cnn`` subdirectory.

   .. code-block:: sh

      cd ./fast_mrf_cnn


2. Open the ``symbol.py`` file and modify the ``decoder_symbol()`` function. You should see the following code there:

   .. code-block:: py

      def decoder_symbol():
          data = mx.sym.Variable('data')
          data = mx.sym.Convolution(data=data, num_filter=256, kernel=(3,3), pad=(1,1), stride=(1, 1), name='deco_conv1')


   Replace the code above with the following:

   .. code-block:: py

      def decoder_symbol_with_vgg(vgg_symbol):
          data = mx.sym.Convolution(data=vgg_symbol, num_filter=256, kernel=(3,3), pad=(1,1), stride=(1, 1), name='deco_conv1')


3. Save and close the ``symbol.py`` file.

4. Open and edit the ``make_image.py`` file. Go to the ``__init__()`` function in the ``Maker`` class:

   .. code-block:: py

      decoder = symbol.decoder_symbol()


   Modify it with the following code:

   .. code-block:: py

      decoder = symbol.decoder_symbol_with_vgg(vgg_symbol)


5. To join the pre-trained weights with the decoder weights, make the following changes:
   After the code lines for loading the decoder weights:

   .. code-block:: py

      args = mx.nd.load('%s_decoder_args.nd'%model_prefix)
      auxs = mx.nd.load('%s_decoder_auxs.nd'%model_prefix)


   Add the following line:

   .. code-block:: py

      arg_dict.update(args)


6. Use ``arg_dict`` instead of ``args`` as a parameter of the ``decoder.bind()`` function. Find the line below:

   .. code-block:: py

      self.deco_executor = decoder.bind(ctx=mx.gpu(), args=args, aux_states=auxs)


   Replace it with the following:

   .. code-block:: py

      self.deco_executor = decoder.bind(ctx=mx.cpu(), args=arg_dict, aux_states=auxs)


7. Add the following code to the end of the ``generate()`` function in the ``Maker`` class to save the result model as a ``.json`` file:

   .. code-block:: py

      self.vgg_executor._symbol.save('{}-symbol.json'.format('vgg19'))
      self.deco_executor._symbol.save('{}-symbol.json'.format('nst_vgg19'))


8. Save and close the ``make_image.py`` file.

**Step 5**: Follow the instructions from the ``README.md`` file in the ``fast_mrf_cnn`` directory of the cloned repository and run the sample with a decoder model.
For example, use the following code to run the sample with the pre-trained decoder weights from the ``models`` folder and output shape:

.. code-block:: py

   import make_image
   maker = make_image.Maker('models/13', (1024, 768))
   maker.generate('output.jpg', '../images/tubingen.jpg')


The ``models/13`` string in the code above is composed of the following substrings:

* ``models/`` -- path to the folder that contains ``.nd`` files with pre-trained styles weights.
* ``13`` -- prefix pointing to the default decoder for the repository, ``13_decoder``.

.. note::

   If an error prompts with "No module named ``cPickle``", try running the script from Step 5 in Python 2. After that return to Python 3 for the remaining steps.

Any style can be selected from `collection of pre-trained weights <https://pan.baidu.com/s/1skMHqYp>`__. On the Chinese-language page, click the down arrow next to a size in megabytes. Then wait for an overlay box to appear, and click the blue button in it to download. The ``generate()`` function generates ``nst_vgg19-symbol.json`` and ``vgg19-symbol.json`` files for the specified shape. In the code, it is ``[1024 x 768]`` for a 4:3 ratio. You can specify another, for example, ``[224,224]`` for a square ratio.

**Step 6**: Run model conversion to generate an Intermediate Representation (IR):

1. Create a new directory. For example:

   .. code-block:: sh

      mkdir nst_model


2. Copy the initial and generated model files to the created directory. For example, to copy the pre-trained decoder weights from the ``models`` folder to the ``nst_model`` directory, run the following commands:

   .. code-block:: sh

      cp nst_vgg19-symbol.json nst_model
      cp vgg19-symbol.json nst_model
      cp ../vgg19.params nst_model/vgg19-0000.params
      cp models/13_decoder_args.nd nst_model
      cp models/13_decoder_auxs.nd nst_model


   .. note::

      Make sure that all the ``.params`` and ``.json`` files are in the same directory as the ``.nd`` files. Otherwise, the conversion process fails.


3. Run model conversion for Apache MXNet. Use the ``--nd_prefix_name`` option to specify the decoder prefix and ``input_shape`` to specify input shapes in ``[N,C,W,H]`` order. For example:

   .. code-block:: sh

      mo --input_symbol <path/to/nst_model>/nst_vgg19-symbol.json --framework mxnet --output_dir <path/to/output_dir> --input_shape [1,3,224,224] --nd_prefix_name 13_decoder --pretrained_model <path/to/nst_model>/vgg19-0000.params


4. The IR is generated (``.bin``, ``.xml`` and ``.mapping`` files) in the specified output directory, and ready to be consumed by the OpenVINO Runtime.

