# Converting a PyTorch Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

@sphinxdirective

The PyTorch framework is supported through export to the ONNX format. In order to optimize and deploy a model that was trained with it:

1. `Export a PyTorch model to ONNX <#exporting-a-pytorch-model-to-onnx-format>`__.
2. :doc:`Convert the ONNX model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX>` to produce an optimized :doc:`Intermediate Representation <openvino_docs_MO_DG_IR_and_opsets>` of the model based on the trained network topology, weights, and biases values.

Exporting a PyTorch Model to ONNX Format
########################################

PyTorch models are defined in Python. To export them, use the ``torch.onnx.export()`` method. The code to
evaluate or test the model is usually provided with its code and can be used for its initialization and export.
The export to ONNX is crucial for this process, but it is covered by PyTorch framework, therefore, It will not be covered here in detail. 
For more information, refer to the `Exporting PyTorch models to ONNX format <https://pytorch.org/docs/stable/onnx.html>`__ guide.

To export a PyTorch model, you need to obtain the model as an instance of ``torch.nn.Module`` class and call the ``export`` function.

.. code-block:: py

   import torch

   # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
   model = SomeModel()
   # Evaluate the model to switch some operations from training mode to inference.
   model.eval()
   # Create dummy input for the model. It will be used to run the model inside export function.
   dummy_input = torch.randn(1, 3, 224, 224)
   # Call the export function
   torch.onnx.export(model, (dummy_input, ), 'model.onnx')


Known Issues
####################

As of version 1.8.1, not all PyTorch operations can be exported to ONNX opset 9 which is used by default.
It is recommended to export models to opset 11 or higher when export to default opset 9 is not working. In that case, use ``opset_version`` option of the ``torch.onnx.export``. For more information about ONNX opset, refer to the `Operator Schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`__ page.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific PyTorch models. Here are some examples:

* :doc:`Convert PyTorch BERT-NER Model <openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Bert_ner>`
* :doc:`Convert PyTorch RCAN Model <openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RCAN>`
* :doc:`Convert PyTorch YOLACT Model <openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_YOLACT>`

@endsphinxdirective
