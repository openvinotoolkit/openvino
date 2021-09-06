# OpenVINO™ LPT: step #1. Prerequisites transformations {#openvino_docs_IE_DG_lpt_step1_prerequisites}

Prerequisites transformations are optional. The goal is prepare a model before to run other low precision transformations. Transformations don't operate with dequantization operations, don't update precisions. Transformations:
* [PullReshapeThroughDequantization](@ref openvino_docs_IE_DG_lpt_PullReshapeThroughDequantization)
* [PullTransposeThroughDequantization](@ref openvino_docs_IE_DG_lpt_PullTransposeThroughDequantization)
* [LinOpSequenceFusion](@ref openvino_docs_IE_DG_lpt_LinOpSequenceFusion)