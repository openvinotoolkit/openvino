Intel® Geti™
===============================


.. meta::
   :description: Intel® Geti™



Intel® Geti™ is an AI platform designed to simplify the development and deployment of computer
vision models. It integrates with the OpenVINO™ to optimize models for efficient
inference on Intel hardware. Intel® Geti™ provides tools for data annotation, model training, 
and deployment, making AI accessible to users of all levels. 


.. image:: ../../../assets/images/geti-optimization-light.svg

Intel Geti Workflow
##########################

1. **Data Collection**: build your dataset for model training. Intel® Geti™ allows annotation 
during upload for classification projects and stores all datasets.

2. **Active Set**: automatically selects optimal media items for training, enhancing efficiency. 
This feature ensures the most relevant data is used to improve model accuracy.

3. **Annotation**: use platform tools to label data, teaching the machine. 
Tools vary by project type, providing flexibility and precision in data labeling.

4. **Training**: Initiates automatically after sufficient annotation. 
Conduct training sessions to refine model predictions, ensuring continuous model improvement.

5. **Optimization and Deployment**: uses OpenVINO to enhance model performance for deployment on Intel hardware. 
OpenVINO optimizes models through techniques like quantization, allowing you to choose precision 
levels such as INT8, FP16, and FP32. This optimization ensures models run 
efficiently across various Intel devices, maximizing inference speed and scalability while maintaining accuracy.
For deployment, models can be served using OpenVINO™ Model Server, facilitating efficient and scalable inference in production environments.

6. **Export**: export models for integration into applications or sharing. 
Seamless export options facilitate collaboration and deployment.

Learn more about Intel® Geti™:

* `Overview <https://docs.geti.intel.com/>`__
* `Documentation <https://docs.geti.intel.com/docs/user-guide/getting-started/introduction>`__
* `Tutorials <https://docs.geti.intel.com/docs/user-guide/getting-started/use-geti/tutorials>`__





