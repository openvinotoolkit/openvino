# Step 1. Prerequisites Transformations {#openvino_docs_OV_UG_lpt_step1_prerequisites}

Prerequisites transformations are optional. The transformations prepare a model before running other low precision transformations. The transformations do not operate with dequantization operations or update precisions. Prerequisites transformations include:
* [PullReshapeThroughDequantization](@ref openvino_docs_OV_UG_lpt_PullReshapeThroughDequantization)
* [PullTransposeThroughDequantization](@ref openvino_docs_OV_UG_lpt_PullTransposeThroughDequantization)
* [LinOpSequenceFusion](@ref openvino_docs_OV_UG_lpt_LinOpSequenceFusion)