# Integrate Inference Engine with Your Python Application {#openvino_docs_IE_DG_integrate_with_customer_application_python}

This document explains how to integrate and use the Inference Engine API with your Python application.   

The following diagram illustrates the typical Inference Engine Python API workflow:
![ie_api_flow_python] 

Read the sections below to learn about each item.

## Link with Inference Engine Library

To make use of the Inference Engine functionality, import IECore to your application: 

```py
from openvino.inference_engine import IECore
``` 
 
## Use Inference Engine API Implement Inference Pipeline

This section provides step-by-step instructions to implement a typical inference pipeline with the Inference Engine Python API:   

![ie_api_use_python]

### Step 1. Create Inference Engine Core

Use the following code to create Inference Engine Core to manage available devices and read network objects: 
```py
ie = IECore()
``` 
### Step 2 (Optional). Read model. Configure Input and Output of the Model

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective

Optionally, configure input and output of the model using the steps below: 

1. Read model 
   @sphinxdirective
      
   .. tab:: IR
   
      .. code-block:: python
   
         net = ie.read_network(model="model.xml")
   
   .. tab:: ONNX
      
      .. code-block:: python
         
         net = ie.read_network(model="model.onnx")
   
   .. tab:: nGraph
      
      .. code-block:: python
         
         // TBD
   
   @endsphinxdirective

2. Request input and output information using input_info, outputs 
   ```py
   input_name = next(iter(net.input_info))  

   output_name = next(iter(net.outputs)) 
   ``` 
   Information for this input layer is stored in input_info. The next cell prints the input layout, precision and shape. 
   ```py
   print(f"input layout: {net.input_info[input_layer].layout}") 
   print(f"input precision: {net.input_info[input_layer].precision}") 
   print(f"input shape: {net.input_info[input_layer].tensor_desc.dims}") 
   ```
   This cell output tells us that the model expects inputs with a shape of [1,3,224,224], and that this is in NCHW layout. This means that the model expects input data with a batch size (N) of 1, 3 channels (C), and images of a height (H) and width (W) of 224. The input data is expected to be of FP32 (floating point) precision. 
    
   Getting the output layout, precision and shape is similar to getting the input layout, precision and shape. 
   ```py
   print(f"output layout: {net.outputs[output_layer].layout}") 
   print(f"output precision: {net.outputs[output_layer].precision}") 
   print(f"output shape: {net.outputs[output_layer].shape}") 
   ```
   This cell output shows that the model returns outputs with a shape of [1, 1001], where 1 is the batch size (N) and 1001 the number of classes (C). The output is returned as 32-bit floating point. 

@sphinxdirective
.. raw:: html

    </div>
@endsphinxdirective 

### Step 3. Load model to the Device 

Load the model to the device using `load_network()`:

@sphinxdirective
   
.. tab:: IR

   .. code-block:: python

      exec_net = ie.load_network(network= "model.xml", device_name="CPU") 
.. tab:: ONNX
   
   .. code-block:: python
      
      exec_net = ie.load_network(network= "model.onnx", device_name="CPU") 

.. tab:: nGraph
   
   .. code-block:: python
      
      // TBD

.. tab:: Model from Step 2
   
   .. code-block:: python
   
      exec_net = ie.load_network(network=net, device_name="CPU")

@endsphinxdirective

### Step 4. Prepare input 
```py
import cv2 
import numpy as np 

image = cv2.imread("image.png") 

# Resize with OpenCV your image if needed to match with net input shape 
# res_image = cv2.resize(src=image, dsize=(W, H)) 

# Converting image to NCHW format with FP32 type 
input_data = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32) 
```

### Step 5. Start Inference
```py
input_name = next(iter(net.input_info))
result = exec_net.infer({input_name: input_data}) 
``` 

### Step 6. Process the Inference Results 
```py
output_name = next(iter(net.outputs))
output = result[output_name] 
```

## Run Application

[ie_api_flow_python]: img/ie_api_python.png
[ie_api_use_python]: img/ie_api_integration_python.png
 