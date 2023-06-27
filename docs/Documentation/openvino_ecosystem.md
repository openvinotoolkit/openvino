# OpenVINO™ Ecosystem Overview {#openvino_ecosystem}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ote_documentation
   datumaro_documentation
   ovsa_get_started
   openvino_docs_tuning_utilities


OpenVINO™ is not just one tool. It is an expansive ecosystem of utilities, providing a comprehensive workflow for deep learning solution development. Learn more about each of them to reach the full potential of OpenVINO™ Toolkit.


**Neural Network Compression Framework (NNCF)**

A suite of advanced algorithms for Neural Network inference optimization with minimal accuracy drop. NNCF applies quantization, filter pruning, binarization and sparsity algorithms to PyTorch and TensorFlow models during training.

More resources:

* :doc:`Documentation <tmo_introduction>`  
* `GitHub <https://github.com/openvinotoolkit/nncf>`__  
* `PyPI <https://pypi.org/project/nncf/>`__  


**OpenVINO™ Training Extensions**

A convenient environment to train Deep Learning models and convert them using the OpenVINO™ toolkit for optimized inference.

More resources:

* :doc:`Overview <ote_documentation>`
* `GitHub <https://github.com/openvinotoolkit/training_extensions>`__
* `Documentation <https://openvinotoolkit.github.io/training_extensions/stable/guide/get_started/introduction.html>`__

**OpenVINO™ Security Add-on**

A solution for Model Developers and Independent Software Vendors to use secure packaging and secure model execution.	 

More resources:

* :doc:`Documentation <ovsa_get_started>`
* `GitHub <https://github.com/openvinotoolkit/security_addon>`__  

**Dataset Management Framework (Datumaro)**

A framework and CLI tool to build, transform, and analyze datasets.

More resources:
 
* :doc:`Overview <datumaro_documentation>`
* `PyPI <https://pypi.org/project/datumaro/>`__  
* `GitHub <https://github.com/openvinotoolkit/datumaro>`__  
* `Documentation <https://openvinotoolkit.github.io/datumaro/stable/docs/get-started/introduction.html>`__ 

**Compile Tool** 


Compile tool is now deprecated. If you need to compile a model for inference on a specific device, use the following script: 

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/export_compiled_model.py
         :language: python
         :fragment: [export_compiled_model]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/export_compiled_model.cpp
         :language: cpp
         :fragment: [export_compiled_model]


To learn which device supports the import / export functionality, see the :doc:`feature support matrix <openvino_docs_OV_UG_Working_with_devices>`.

For more details on preprocessing steps, refer to the :doc:`Optimize Preprocessing <openvino_docs_OV_UG_Preprocessing_Overview>`. To compile the model with advanced preprocessing capabilities, refer to the :doc:`Use Case - Integrate and Save Preprocessing Steps Into OpenVINO IR <openvino_docs_OV_UG_Preprocess_Usecase_save>`, which shows how to have all the preprocessing in the compiled blob. 

**DL Workbench**

A web-based tool for deploying deep learning models. Built on the core of OpenVINO and equipped with a graphics user interface, DL Workbench is a great way to explore the possibilities of the OpenVINO workflow, import, analyze, optimize, and build your pre-trained models. You can do all that by visiting `Intel® Developer Cloud <https://software.intel.com/content/www/us/en/develop/tools/devcloud.html>`__ and launching DL Workbench online.

**OpenVINO™ integration with TensorFlow (OVTF)**

OpenVINO™ Integration with TensorFlow will no longer be supported as of OpenVINO release 2023.0. As part of the 2023.0 release, OpenVINO will feature a significantly enhanced TensorFlow user experience within native OpenVINO without needing offline model conversions. :doc:`Learn more <openvino_docs_MO_DG_TensorFlow_Frontend>`.


@endsphinxdirective

