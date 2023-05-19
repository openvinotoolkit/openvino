# Running and Deploying Inference {#openvino_docs_deployment_guide_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Run and Deploy Locally <openvino_deployment_guide>
   Deploy via Model Serving <ovms_what_is_openvino_model_server>


Once you have a model that meets both OpenVINO™ and your requirements, you can choose how to deploy it with your application.

.. panels::

    :doc:`Deploy via OpenVINO Runtime <openvino_deployment_guide>` 
    ^^^^^^^^^^^^^^

    Local deployment uses OpenVINO Runtime that is called from, and linked to, the application directly. 
    It utilizes resources available to the system and provides the quickest way of launching inference.
    ---

    :doc:`Deploy via Model Server <ovms_what_is_openvino_model_server>`
    ^^^^^^^^^^^^^^

    Deployment via OpenVINO Model Server allows the application to connect to the inference server set up remotely. 
    This way inference can use external resources instead of those available to the application itself. 


Apart from the default deployment options, you may also :doc:`deploy your application for the TensorFlow framework with OpenVINO Integration <ovtf_integration>`

@endsphinxdirective