// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_model_to_fp16_element_type.hpp"

#include "itt.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/common_optimizations/mark_shape_subgraphs.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::ConvertModelToFP16ElementType::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertModelToFP16ElementType);
    if (!ngraph::op::util::has_decompression_converts(f))
        return false;

    Manager manager(get_pass_config());
    // Mark nodes in ShapeOf subgraphs with disable_fp16_compression rt_info to keep them in FP32 precision
    if (m_keep_precision_sensitive_in_fp32)
        REGISTER_PASS(manager, MarkEntireShapeSubgraphs)

    const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
    type_to_fuse_map additional_fuse_map = {};
    REGISTER_PASS(manager,
                  ConvertPrecision,
                  convert_precision_list,
                  additional_fuse_map,
                  m_keep_precision_sensitive_in_fp32);

    REGISTER_PASS(manager, EnableDecompressionConvertConstantFolding)
    REGISTER_PASS(manager, ConstantFolding)
    manager.run_passes(f);

    return false;
}
