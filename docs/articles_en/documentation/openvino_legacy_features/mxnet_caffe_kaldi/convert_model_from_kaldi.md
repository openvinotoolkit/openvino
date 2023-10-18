# Converting a Kaldi Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from the 
                 Kaldi format to the OpenVINO Intermediate Representation.


.. warning::

   Note that OpenVINO support for Kaldi is currently being deprecated and will be removed entirely in the future.

.. note::

   Model conversion API supports the `nnet1 <http://kaldi-asr.org/doc/dnn1.html>`__ and `nnet2 <http://kaldi-asr.org/doc/dnn2.html>`__ formats of Kaldi models. The support of the `nnet3 <http://kaldi-asr.org/doc/dnn3.html>`__ format is limited.

To convert a Kaldi model, run model conversion with the path to the input model ``.nnet`` or ``.mdl`` file:

.. code-block:: cpp

   mo --input_model <INPUT_MODEL>.nnet

Using Kaldi-Specific Conversion Parameters
##########################################

The following list provides the Kaldi-specific parameters.

.. code-block:: cpp

   Kaldi-specific parameters:
   --counts COUNTS       A file name with full path to the counts file or empty string to utilize count values from the model file
   --remove_output_softmax
                           Removes the Softmax that is the output layer
   --remove_memory       Remove the Memory layer and add new inputs and outputs instead

Examples of CLI Commands
########################

* To launch model conversion for the ``wsj_dnn5b_smbr`` model with the specified ``.nnet`` file:

  .. code-block:: cpp

    mo --input_model wsj_dnn5b_smbr.nnet

* To launch model conversion for the ``wsj_dnn5b_smbr`` model with the existing file that contains counts for the last layer with biases:

  .. code-block:: cpp

    mo --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts


  * The model conversion normalizes сounts in the following way:

    .. math::

       S = \frac{1}{\sum_{j = 0}^{|C|}C_{j}}

    .. math::

       C_{i}=log(S*C_{i})

    where :math:`C` - the counts array, :math:`C_{i} - i^{th}` element of the counts array, :math:`|C|` - number of elements in the counts array;

  * The normalized counts are subtracted from biases of the last or next to last layer (if last layer is SoftMax).

    .. note:: Model conversion API will show a warning if a model contains values of counts and the ``counts`` option is not used.

* If you want to remove the last SoftMax layer in the topology, launch the model conversion with the ``remove_output_softmax`` flag:

  .. code-block:: cpp

     mo --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts --remove_output_softmax

  Model conversion API finds the last layer of the topology and removes this layer only if it is a SoftMax layer.

  .. note:: Model conversion can remove SoftMax layer only if the topology has one output.

* You can use the *OpenVINO Speech Recognition* sample application for the sample inference of Kaldi models. This sample supports models with only one output. If your model has several outputs, specify the desired one with the ``output`` option.

Supported Kaldi Layers
######################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations_frontend>` page.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific Kaldi models. Here are some examples:

* :doc:`Convert Kaldi ASpIRE Chain Time Delay Neural Network (TDNN) Model <openvino_docs_MO_DG_prepare_model_convert_model_kaldi_specific_Aspire_Tdnn_Model>`


@endsphinxdirective

