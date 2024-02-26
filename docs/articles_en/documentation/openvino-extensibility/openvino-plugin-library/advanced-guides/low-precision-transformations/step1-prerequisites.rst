.. {#../low-precision-transformations_step1_prerequisites}

Step 1. Prerequisites Transformations
=====================================


.. meta::
   :description: Learn about optional Prerequisites transformations, that 
                 prepare a model before applying other low precision transformations.

.. toctree::
   :maxdepth: 1
   :hidden:

   PullReshapeThroughDequantization <../low-precision-transformations_PullReshapeThroughDequantization>
   PullTransposeThroughDequantization <../low-precision-transformations_PullTransposeThroughDequantization>
   LinOpSequenceFusion <../low-precision-transformations_LinOpSequenceFusion>

Prerequisites transformations are optional. The transformations prepare a model before running other low precision transformations. The transformations do not operate with dequantization operations or update precisions. Prerequisites transformations include:

* :doc:`PullReshapeThroughDequantization <../low-precision-transformations_PullReshapeThroughDequantization>`
* :doc:`PullTransposeThroughDequantization <../low-precision-transformations_PullTransposeThroughDequantization>`
* :doc:`LinOpSequenceFusion <../low-precision-transformations_LinOpSequenceFusion>`

