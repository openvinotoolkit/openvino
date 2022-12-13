# OpenVINO™ Deployment {#openvino_docs_deployment_guide_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Run Inference <openvino_docs_OV_UG_OV_Runtime_User_Guide>
   Inference Optimization <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>

.. toctree::
   :maxdepth: 1
   :hidden:

   Deploy Locally <openvino_deployment_guide>
   Deploy Using Model Server <ovms_what_is_openvino_model_server>


Once you have a model that meets both OpenVINO™ and your requirements, you can choose among several ways of deploying it with your application. The two default options are:

.. panels::

    `Deploying locally <openvino_deployment_guide>`_ 
    ^^^^^^^^^^^^^^

    Local deployment simply uses OpenVINO Runtime installed on the device. It utilizes resources available to the system.

    ---

    `Deploying by Model Serving <ovms_what_is_openvino_model_server>`_
    ^^^^^^^^^^^^^^

    Deployment via OpenVINO Model Server allows the device to connect to the server set up remotely. This way inference uses external resources instead of the ones provided by the device itself. 

@endsphinxdirective


> **NOTE**: Note that [running inference in OpenVINO Runtime](../OV_Runtime_UG/openvino_intro.md) is the most basic form of deployment. Before moving forward, make sure you know how to create a proper Inference configuration. Inference may be additionally optimized, as described in the [Inference Optimization section](../optimization_guide/dldt_deployment_optimization_guide.md).

Apart from the default deployment options, you may also [deploy your application for the TensorFlow framework with OpenVINO Integration](./openvino_ecosystem_ovtf.md).