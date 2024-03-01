Python Wheels in OpenVINO™ Toolkit
===================================

Introduction
#############
Python wheels are a distribution format that allows for faster and more efficient installation of Python packages.
In the context of OpenVINO™ Toolkit, Python wheels provide a convenient way to work with the toolkit in your Python projects.
This section will guide you through the usage of Python wheels in OpenVINO™.

What are Python Wheels?
########################
Python wheels are binary distribution formats that aim to make the installation of Python packages faster and more reliable.
In the case of OpenVINO™, wheels are particularly useful for quick and hassle-free deployment. You can find OpenVINO™ Python distribution on PyPI: `OpenVINO on PyPI. <https://pypi.org/project/openvino/#files>`__

Available Wheels in OpenVINO™
++++++++++++++++++++++++++++++
OpenVINO™ offers two main wheels:
  
`openvino <https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python/wheel/setup.py>`__: Core Python bindings for OpenVINO.

`openvino-dev <https://github.com/openvinotoolkit/openvino/blob/master/tools/openvino_dev/setup.py>`__: Additional development tools and utilities.

Building Python Wheels
#######################
To build Python wheels during the project build process, make sure to include the following options when running the cmake command:
  1. bash
  2. Copy code
  3. cmake -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON ..

This ensures that the build process includes the Python components and generates the necessary wheel files.

Locating Python Wheels
#######################
After building the project, you can find the generated Python wheels under the **openvino_install_dir/tools** directory.
This is the default location where the wheels are stored for easy access.

Development with Python Wheels
###############################
When using OpenVINO™ Python wheels, there are some differences in the development workflow compared to setting the PYTHONPATH to OpenVINO directories.

Python Environment Setup
+++++++++++++++++++++++++
Ensure that you have the necessary Python wheels installed in your environment:
  1. bash
  2. Copy code
  3. pip install openvino
  4. pip install openvino-dev

Workflow Differences
+++++++++++++++++++++
Importing Modules: With wheels, you can directly import OpenVINO modules in your Python scripts without setting PYTHONPATH.
  1. python
  2. Copy code
  3. from openvino import inference_engine as ie
*Development Isolation*:
  Python wheels offer a more isolated development environment, reducing potential conflicts with other Python projects.
*Simplified Dependency Management*:
  Python wheels encapsulate the required dependencies, simplifying the setup for your Python projects.
*Version Compatibility*:
  Ensure that the installed wheels match your project's requirements and dependencies.

Conclusion
###########
Integrating OpenVINO™ Toolkit into your Python projects is made easier with the use of Python wheels.
They provide a streamlined development experience, simplifying the deployment and usage of OpenVINO functionality in your applications.
Explore the possibilities and optimize your AI inference workflow with OpenVINO™ Python wheels!