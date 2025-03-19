Most Efficient Large Language Models for AI PC
==============================================

This page is regularly updated to help you identify the best-performing LLMs on the
Intel® Core™ Ultra processor family and AI PCs.
The current data is as of OpenVINO 2025.0, 06 March 2025 (7-155H and 7-268V)
and OpenVINO 2024.6, 13 Dec. 2024 (9-288V).

The tables below list the key performance indicators for inference on built-in GPUs.



.. tab-set::

   .. tab-item:: 9-288V

      .. data-table::
         :class: modeldata stripe
         :name: supportedModelsTable_V1
         :header-rows: 1
         :file:  ../../_static/benchmarks_files/llm_models_9-288V.csv
         :data-column-hidden: [3,4,6]
         :data-order: [[ 0, "asc" ]]
         :data-page-length: 10

   .. tab-item:: 7-268V

      .. data-table::
         :class: modeldata stripe
         :name: supportedModelsTable_V2
         :header-rows: 1
         :file:  ../../_static/benchmarks_files/llm_models_7-258V.csv
         :data-column-hidden: [3,4,6]
         :data-order: [[ 0, "asc" ]]

   .. tab-item:: 7-155H

      .. data-table::
         :class: modeldata stripe
         :name: supportedModelsTable_V3
         :header-rows: 1
         :file:  ../../_static/benchmarks_files/llm_models_7-155H.csv
         :data-column-hidden: [3,4,6]
         :data-order: [[ 0, "asc" ]]


.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      All models listed here were tested with the following parameters:

      *  Framework: PyTorch
      *  Beam: 1
      *  Batch size: 1

   .. grid-item::

      .. button-link:: https://docs.openvino.ai/2025/_static/download/benchmarking_genai_platform_list.pdf
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Get system descriptions [PDF]

      .. button-link:: ../../_static/benchmarks_files/llm_models.csv
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Get the data in .csv [CSV]

