.. {#openvino_docs_OV_UG_lpt_step1_prerequisites}

Step 1. Prerequisites Transformations
=====================================


.. meta::
   :description: Learn about optional Prerequisites transformations, that 
                 prepare a model before applying other low precision transformations.

.. toctree::
   :maxdepth: 1
   :hidden:

   PullReshapeThroughDequantization <openvino_docs_OV_UG_lpt_PullReshapeThroughDequantization>
   PullTransposeThroughDequantization <openvino_docs_OV_UG_lpt_PullTransposeThroughDequantization>
   LinOpSequenceFusion <openvino_docs_OV_UG_lpt_LinOpSequenceFusion>

Prerequisites transformations are optional. The transformations prepare a model before running other low precision transformations. The transformations do not operate with dequantization operations or update precisions. Prerequisites transformations include:

* :doc:`PullReshapeThroughDequantization <openvino_docs_OV_UG_lpt_PullReshapeThroughDequantization>`
* :doc:`PullTransposeThroughDequantization <openvino_docs_OV_UG_lpt_PullTransposeThroughDequantization>`
* :doc:`LinOpSequenceFusion <openvino_docs_OV_UG_lpt_LinOpSequenceFusion>`

