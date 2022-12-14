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

    `Deploy Locally <openvino_deployment_guide>`_ 
    ^^^^^^^^^^^^^^

    Local deployment uses OpenVINO Runtime installed on the device. It utilizes resources available to the system and provides the quickest way of launching inference.

    ---

    `Deploy by Model Serving <ovms_what_is_openvino_model_server>`_
    ^^^^^^^^^^^^^^

    Deployment via OpenVINO Model Server allows the device to connect to the server set up remotely. This way inference uses external resources instead of the ones provided by the device itself. 

@endsphinxdirective



Apart from the default deployment options, you may also [deploy your application for the TensorFlow framework with OpenVINO Integration](./openvino_ecosystem_ovtf.md).