Converting a TensorFlow CRNN Model
==================================


.. meta::
   :description: Learn how to convert a CRNN model
                 from TensorFlow to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

This tutorial explains how to convert a CRNN model to OpenVINO™ Intermediate Representation (IR).

There are several public versions of TensorFlow CRNN model implementation available on GitHub. This tutorial explains how to convert the model from
the `CRNN Tensorflow <https://github.com/MaybeShewill-CV/CRNN_Tensorflow>`__ repository to IR, and is validated with Python 3.7, TensorFlow 1.15.0, and protobuf 3.19.0.
If you have another implementation of CRNN model, it can be converted to OpenVINO IR in a similar way. You need to get inference graph and run model conversion of it.

**To convert the model to IR:**

**Step 1.** Clone this GitHub repository and check out the commit:

1. Clone the repository:

   .. code-block:: sh

      git clone https://github.com/MaybeShewill-CV/CRNN_Tensorflow.git

2. Go to the ``CRNN_Tensorflow`` directory of the cloned repository:

   .. code-block:: sh

      cd path/to/CRNN_Tensorflow

3. Check out the necessary commit:

   .. code-block:: sh

      git checkout 64f1f1867bffaacfeacc7a80eebf5834a5726122


**Step 2.** Train the model using the framework or the pretrained checkpoint provided in this repository.


**Step 3.** Create an inference graph:

1. Add the ``CRNN_Tensorflow`` folder to ``PYTHONPATH``.

   * For Linux:

     .. code-block:: sh

        export PYTHONPATH="${PYTHONPATH}:/path/to/CRNN_Tensorflow/"


   * For  Windows, add ``/path/to/CRNN_Tensorflow/`` to the ``PYTHONPATH`` environment variable in settings.

2. Edit the ``tools/demo_shadownet.py`` script. After ``saver.restore(sess=sess, save_path=weights_path)`` line, add the following code:

   .. code-block:: py
      :force:

      from tensorflow.python.framework import graph_io
      frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['shadow/LSTMLayers/transpose_time_major'])
      graph_io.write_graph(frozen, '.', 'frozen_graph.pb', as_text=False)

3. Run the demo with the following command:

   .. code-block:: sh

      python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-10-17-11-47-46.ckpt-199999


   If you want to use your checkpoint, replace the path in the ``--weights_path`` parameter with a path to your checkpoint.

4. In the ``CRNN_Tensorflow`` directory, you will find the inference CRNN graph ``frozen_graph.pb``. You can use this graph with OpenVINO to convert the model to IR and then run inference.

**Step 4.** Convert the model to IR:

.. code-block:: sh

   mo --input_model path/to/your/CRNN_Tensorflow/frozen_graph.pb

