// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/non_max_suppression.hpp"
#include <ov_ops/nms_ie_internal.hpp>

#include "intel_gpu/primitives/non_max_suppression.hpp"

namespace ov {
namespace intel_gpu {

static void CreateNonMaxSuppressionIEInternalOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& op) {
    cldnn::non_max_suppression::Rotation rotation = cldnn::non_max_suppression::Rotation::NONE;
    const bool is_nms_rotated = op->m_rotation != ov::op::internal::NonMaxSuppressionIEInternal::Rotation_None;
    if (is_nms_rotated) {
        // For NMSRotated threshold inputs are mandatory, and soft_nms_sigma input is absent
        validate_inputs_count(op, {5});

        rotation = op->m_rotation == ov::op::internal::NonMaxSuppressionIEInternal::Rotation_Clockwise ?
                    cldnn::non_max_suppression::Rotation::CLOCKWISE
                    : cldnn::non_max_suppression::Rotation::COUNTERCLOCKWISE;
    } else {
        validate_inputs_count(op, {2, 3, 4, 5, 6});
    }
    auto inputs = p.GetInputInfo(op);

    auto boxesShape = op->get_input_partial_shape(0);
    size_t num_outputs = op->get_output_size();
    auto nonMaxSuppressionLayerName = layer_type_name_ID(op);
    auto prim = cldnn::non_max_suppression(
            nonMaxSuppressionLayerName,
            inputs,
            0,
            op->m_center_point_box,
            op->m_sort_result_descending,
            num_outputs);

    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});
    prim.rotation = rotation;

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, NonMaxSuppressionIEInternal);

}  // namespace intel_gpu
}  // namespace ov
