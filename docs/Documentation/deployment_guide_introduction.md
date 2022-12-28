# OpenVINO™ Deployment {#openvino_docs_deployment_guide_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Deploy via OpenVINO Runtime <openvino_deployment_guide>
   Deploy via Model Serving <ovms_what_is_openvino_model_server>

@endsphinxdirective


Once you have a model that meets both OpenVINO™ and your requirements, you can choose how to deploy it with your application.

@sphinxdirective
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

@endsphinxdirective



Apart from the default deployment options, you may also [deploy your application for the TensorFlow framework with OpenVINO Integration](./openvino_ecosystem_ovtf.md).
