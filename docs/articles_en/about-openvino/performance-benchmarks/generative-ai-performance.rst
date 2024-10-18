Most Efficient Large Language Models for AI PC
==============================================

This page is regularly updated to help you identify the best-performing LLMs on the
Intel® Core™ Ultra processor family and AI PCs.

The tables below list key performance indicators for a selection of Large Language Models,
running on an Intel® Core™ Ultra 7-165H based system, on built-in GPUs.



.. raw:: html

   <label><link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css"></label>



.. tab-set::

   .. tab-item:: OpenVINO

      .. csv-table::
         :class: modeldata stripe
         :name: supportedModelsTableOv
         :header-rows: 1
         :file:  ../../_static/benchmarks_files/llm_models.csv


.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      All models listed here were tested with the following parameters:

      *  Framework: PyTorch
      *  Model precision: INT4
      *  Beam: 1
      *  Batch size: 1

   .. grid-item::

      .. button-link:: https://docs.openvino.ai/2024/_static/benchmarks_files/OV-2024.4-platform_list.pdf
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Get full system info [PDF]

      .. button-link:: ../../_static/benchmarks_files/llm_models.csv
         :color: primary
         :outline:
         :expand:

         :material-regular:`download;1.5em` Get the data in .csv [CSV]

