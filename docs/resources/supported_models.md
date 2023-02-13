# Supported Models {#openvino_supported_models}


The OpenVINO team continues the effort to support as many models out-of-the-box as possible. 
Based on our research and user feedback, we prioritize the most common models and test them 
before every release. These models are considered officially supported.

@sphinxdirective

.. button-link:: _static/download/OV_2022_models_supported.pdf
   :color: primary
   :outline:

   :material-regular:`download;1.5em` Click for supported models [PDF]


| Note that the list provided here does not include all models supported by OpenVINO.
| If your model is not included but is similar to those that are, it is still very likely to work. 
  If your model fails to execute properly there are a few options available: 

@endsphinxdirective

* If the model originates from a framework like TensorFlow or PyTorch, OpenVINO™ offers a hybrid solution. The original model can be run without explicit conversion into the OpenVINO format. For more information, see [OpenVINO TensorFlow Integration](https://docs.openvino.ai/latest/ovtf_integration.html).  
* You can create a GitHub request for the operation(s) that are missing. These requests are reviewed regularly. You will be informed if and how the request will be accommodated. Additionally, your request may trigger a reply from someone in the community who can help.  
* As OpenVINO™ is open source you can enhance it with your own contribution to the GitHub repository. To learn more, see the articles on [OpenVINO Extensibility](https://docs.openvino.ai/latest/openvino_docs_Extensibility_UG_Intro.html).


The following table summarizes the number of models supported by OpenVINO™ in different categories:

@sphinxdirective
+--------------------------------------------+-------------------+
| Model Categories:                          | Number of Models: |
+============================================+===================+
| Object Detection	                         | 149	             |
+--------------------------------------------+-------------------+
| Instance Segmentation                      | 3                 |
+--------------------------------------------+-------------------+
| Semantic Segmentation                      | 19                |
+--------------------------------------------+-------------------+
| Image Processing, Enhancement	             | 16                |
+--------------------------------------------+-------------------+
| Monodepth	                                 | 2                 |
+--------------------------------------------+-------------------+
| Colorization	                             | 2                 |
+--------------------------------------------+-------------------+
| Behavior / Decision Prediction	         | 1                 |
+--------------------------------------------+-------------------+
| Action Recognition	                     | 2                 |
+--------------------------------------------+-------------------+
| Time Series Forecasting	                 | 1                 |
+--------------------------------------------+-------------------+
| Image Classification                       | 68                |
+--------------------------------------------+-------------------+
| Image Classification, Dual Path Network    | 1                 |
+--------------------------------------------+-------------------+
| Image Classification, Emotion              | 1                 |
+--------------------------------------------+-------------------+
| Image Translation	                         | 1                 |
+--------------------------------------------+-------------------+
| Natural language Processing	             | 35                |
+--------------------------------------------+-------------------+
| Text Detection	                         | 18                |
+--------------------------------------------+-------------------+
| Audio Enhancement	                         | 3                 |
+--------------------------------------------+-------------------+
| Sound Classification	                     | 2                 |
+--------------------------------------------+-------------------+
@endsphinxdirective