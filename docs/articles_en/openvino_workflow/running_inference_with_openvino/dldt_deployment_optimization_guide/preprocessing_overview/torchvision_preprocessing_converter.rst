.. {#torchvision_preprocessing_converter}

Torchvision preprocessing converter
=======================================


.. meta::
   :description: See how OpenVINO™ enables torchvision preprocessing
                 to optimize model inference.


The Torchvision-to-OpenVINO converter enables automatic translation of operators from the torchvision
preprocessing pipeline to the OpenVINO format and embed them in your model. It is often used to adjust
images serving as input for AI models to have proper dimensions or data types.

As the converter is fully based on the **openvino.preprocess** module, you can implement the **torchvision.transforms**
feature easily and without the use of external libraries, reducing the overall application complexity
and enabling additional performance optimizations.


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


Example
###################

.. code-block:: py

   preprocess_pipeline = torchvision.transforms.Compose(
       [
           torchvision.transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
           torchvision.transforms.CenterCrop((216, 218)),
           torchvision.transforms.Pad((2, 3, 4, 5), fill=3),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.ConvertImageDtype(torch.float32),
           torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ]
   )

   torch_model = SimpleConvnet(input_channels=3)

   torch.onnx.export(torch_model, torch.randn(1, 3, 224, 224), "test_convnet.onnx", verbose=False, input_names=["input"], output_names=["output"])
   core = Core()
   ov_model = core.read_model(model="test_convnet.onnx")

   test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)
   ov_model = PreprocessConverter.from_torchvision(
       model=ov_model, transform=preprocess_pipeline, input_example=Image.fromarray(test_input.astype("uint8"), "RGB")
   )
   ov_model = core.compile_model(ov_model, "CPU")
   ov_input = np.expand_dims(test_input, axis=0)
   output = ov_model.output(0)
   ov_result = ov_model(ov_input)[output]





