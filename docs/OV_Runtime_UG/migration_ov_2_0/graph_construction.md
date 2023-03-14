# Model Creation in OpenVINO™ Runtime {#openvino_2_0_model_creation}

@sphinxdirective

OpenVINO™ Runtime with API 2.0 includes the nGraph engine as a common part. The ``ngraph`` namespace has been changed to ``ov``, but all other parts of the ngraph API have been preserved.

The code snippets below show how to change the application code for migration to API 2.0.

nGraph API
####################

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ngraph.cpp
         :language: cpp
         :fragment: ngraph:graph

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ngraph.py
         :language: Python
         :fragment: ngraph:graph


API 2.0
####################


.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_graph.cpp
         :language: cpp
         :fragment: ov:graph

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_graph.py
         :language: Python
         :fragment: ov:graph


Additional Resources
####################

* :doc:`Hello Model Creation C++ Sample <openvino_inference_engine_samples_model_creation_sample_README>`
* :doc:`Hello Model Creation Python Sample <openvino_inference_engine_ie_bridges_python_sample_model_creation_sample_README>`

@endsphinxdirective
