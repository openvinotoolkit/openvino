Install Intel® Distribution of OpenVINO™ Toolkit from npm Registry
==================================================================

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and
                 macOS operating systems, using the npm registry.


.. note::

   Note that the npm distribution:

   * offers the JavaScript API only
   * is dedicated to users of all major OSes: Windows, Linux, and macOS
     (all x86_64 / arm64 architectures)
   * macOS offers support only for CPU inference

   Before installing OpenVINO, see the
   :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.

Installing OpenVINO Node.js
###########################

1. Make sure that you have installed `Node.js and npm <https://nodejs.org/en/download>`__
   on your system.
2. Navigate to your project directory and run the following command in the terminal:

   .. code-block:: sh

      npm install openvino-node

.. note::

   The *openvino-node* npm package runs in Node.js environment only and provides
   a subset of `OpenVINO Runtime C++ API <https://docs.openvino.ai/2025/api/c_cpp_api/group__ov__cpp__api.html>`__.

What's Next?
####################

Now that you’ve installed OpenVINO npm package, you’re ready to run your own machine
learning applications! Explore :doc:`OpenVINO Node.js API <../../api/nodejs_api/nodejs_api>`
to learn more about how to integrate a model in Node.js applications.

Additional Resources
####################

- Intel® Distribution of OpenVINO™ toolkit home page: https://software.intel.com/en-us/openvino-toolkit
