OpenVINO™ Node.js Bindings
==========================

.. meta::
   :description: Explore Node.js API and implementation of its features in Intel®
                 Distribution of OpenVINO™ Toolkit.

.. toctree::
   :maxdepth: 3
   :hidden:

   addon <./addon>

Use OpenVINO JavaScript API for your Node.js application.

Usage
#####################

1. Install the **openvino-node** package:

   .. code-block::

      npm install openvino-node

2. Use the **openvino-node** package:

   .. code-block::

      const { addon: ov } = require('openvino-node');


Build From Sources
#####################

For more details, refer to the
`OpenVINO™ JavaScript API Developer Documentation
<https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation>`__



API Development
#####################

Get started with OpenVINO Node.js API development :doc:`here <./addon>`.

With OpenVINO 2024.4, the following methods have been introduced:

- :ref:`Model.clone() <clone>`
- :ref:`Model.getOutputElementType() <getOutputElementType>`
- :ref:`CompiledModel.getProperty() <getProperty>`
- :ref:`CompiledModel.setProperty() <setProperty>`


Additional Resources
#####################

- `OpenVINO™ Node.js Bindings Examples of Usage <https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/README.md>`__
- `OpenVINO™ Core Components <https://github.com/openvinotoolkit/openvino/blob/master/src/README.md>`__
- `OpenVINO™ Python API <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/README.md>`__
- `OpenVINO™ Other Bindings <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/README.md>`__