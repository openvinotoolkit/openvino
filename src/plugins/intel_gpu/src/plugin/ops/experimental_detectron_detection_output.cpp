// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_detection_output.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/primitives/experimental_detectron_detection_output.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateExperimentalDetectronDetectionOutputOp(
    Program& p,
    const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>& op) {
    p.ValidateInputs(op, {4});

    if (op->get_output_size() != 3) {
        IE_THROW() << "ExperimentalDetectronDetectionOutput requires 3 outputs";
    }

    auto inputs = p.GetInputPrimitiveIDs(op);

    const auto& attrs = op->get_attrs();

    const auto op_friendly_name = op->get_friendly_name();

    const auto layer_type_name = layer_type_name_ID(op);
    const auto layer_name = layer_type_name + ".0";

    const auto mutable_precision1 = op->get_output_element_type(1);
    const auto output_shape1 = op->get_output_shape(1);
    const cldnn::layout mutable_layout1{DataTypeFromPrecision(mutable_precision1),
                                        DefaultFormatForDims(output_shape1.size()),
                                        tensor_from_dims(output_shape1)};
    cldnn::memory::ptr shared_memory1{p.GetEngine().allocate_memory(mutable_layout1)};

    const auto mutable_id_w1 = layer_type_name + "_md_write.1";
    const cldnn::mutable_data mutable_prim_w{mutable_id_w1, shared_memory1, op_friendly_name};
    p.primitiveIDs[mutable_id_w1] = mutable_id_w1;
    p.AddPrimitive(mutable_prim_w);
    inputs.push_back(mutable_id_w1);

    const auto mutable_precision2 = op->get_output_element_type(2);
    const auto output_shape2 = op->get_output_shape(2);
    const cldnn::layout mutable_layout2{DataTypeFromPrecision(mutable_precision2),
                                        DefaultFormatForDims(output_shape2.size()),
                                        tensor_from_dims(output_shape2)};
    cldnn::memory::ptr shared_memory2{p.GetEngine().allocate_memory(mutable_layout2)};

    const auto mutable_id_w2 = layer_type_name + "_md_write.2";
    const cldnn::mutable_data mutable_prim_w2{mutable_id_w2, shared_memory2, op_friendly_name};
    p.primitiveIDs[mutable_id_w2] = mutable_id_w2;
    p.AddPrimitive(mutable_prim_w2);
    inputs.push_back(mutable_id_w2);

    const auto expectedPrimInputCount = 4 + 2; // 4 operation inputs plus 2 input-outputs
    if (inputs.size() != expectedPrimInputCount) {
        IE_THROW() << "experimental_detectron_detection_output primitive requires 6 inputs";
    }

    const cldnn::experimental_detectron_detection_output prim{layer_name,
                                                              inputs[0],
                                                              inputs[1],
                                                              inputs[2],
                                                              inputs[3],
                                                              inputs[4],  // output classes
                                                              inputs[5],  // output scores
                                                              attrs.score_threshold,
                                                              attrs.nms_threshold,
                                                              static_cast<int>(attrs.num_classes),
                                                              static_cast<int>(attrs.post_nms_count),
                                                              static_cast<int>(attrs.max_detections_per_image),
                                                              attrs.class_agnostic_box_regression,
                                                              attrs.max_delta_log_wh,
                                                              attrs.deltas_weights,
                                                              op_friendly_name};

    p.AddPrimitive(prim);

    const auto mutable_id_r1 = layer_type_name + ".1";
    const cldnn::mutable_data mutable_prim_r1{mutable_id_r1, {layer_name}, shared_memory1, op_friendly_name};
    p.primitiveIDs[mutable_id_r1] = mutable_id_r1;
    p.AddPrimitive(mutable_prim_r1);

    const auto mutable_id_r2 = layer_type_name + ".2";
    const cldnn::mutable_data mutable_prim_r2{mutable_id_r2, {layer_name}, shared_memory2, op_friendly_name};
    p.primitiveIDs[mutable_id_r2] = mutable_id_r2;
    p.AddPrimitive(mutable_prim_r2);

    p.AddPrimitiveToProfiler(prim, op);
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronDetectionOutput);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
