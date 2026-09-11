Torchvision Preprocessing Converter
=======================================


.. meta::
   :description: See how OpenVINO™ enables torchvision preprocessing
                 to optimize model inference.


The Torchvision-to-OpenVINO converter enables automatic translation of operators from the
torchvision preprocessing pipeline to the OpenVINO format and embed them in your model. It is
often used to adjust images serving as input for AI models to have proper dimensions or data
types.

As the converter is fully based on the **openvino.preprocess** module, you can implement the
**torchvision.transforms** feature easily and without the use of external libraries, reducing
the overall application complexity and enabling additional performance optimizations.


.. note::

   Not all torchvision transforms are supported yet. The following operations are available:

   .. code-block::

      transforms.Compose
      transforms.Normalize
      transforms.ConvertImageDtype
      transforms.Grayscale
      transforms.Pad
      transforms.ToTensor
      transforms.CenterCrop
      transforms.Resize


.. note::

   ``transforms.Resize`` with ``InterpolationMode.NEAREST`` is implemented on top of the
   ``Interpolate`` operator's ``nearest`` mode, configured with ``coordinate_transformation_mode``
   and ``nearest_mode`` attributes chosen to closely mirror Pillow's rounding behavior. This
   configuration is not guaranteed to be bit-exact with ``torchvision``/Pillow for every
   input/output size combination, especially for non-square images where the height and width
   scale factors differ. See the ``nearest_mode`` note in the
   :doc:`Interpolate <../../../../documentation/openvino-ir-format/operation-sets/operation-specs/image/interpolate-11>`
   operation specification for details and possible future improvements.


Example
###################

.. doxygensnippet:: docs/articles_en/assets/snippets/torchvision_preprocessing.py
    :language: Python
    :fragment: torchvision_preprocessing
