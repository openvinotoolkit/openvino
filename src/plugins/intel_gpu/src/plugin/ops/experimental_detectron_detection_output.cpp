// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_detection_output.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/experimental_detectron_detection_output.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov::intel_gpu {

static void CreateExperimentalDetectronDetectionOutputOp(
    ProgramBuilder& p,
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>& op) {
    validate_inputs_count(op, {4});

    if (op->get_output_size() != 3) {
        OPENVINO_THROW("ExperimentalDetectronDetectionOutput requires 3 outputs");
    }

    auto inputs = p.GetInputInfo(op);

    const auto& attrs = op->get_attrs();

    if (p.use_new_shape_infer()) {
        cldnn::experimental_detectron_detection_output prim{layer_type_name_ID(op),
                                                            inputs[0],
                                                            inputs[1],
                                                            inputs[2],
                                                            inputs[3],
                                                            attrs.score_threshold,
                                                            attrs.nms_threshold,
                                                            static_cast<int>(attrs.num_classes),
                                                            static_cast<int>(attrs.post_nms_count),
                                                            static_cast<int>(attrs.max_detections_per_image),
                                                            attrs.class_agnostic_box_regression,
                                                            attrs.max_delta_log_wh,
                                                            attrs.deltas_weights};
        prim.num_outputs = op->get_output_size();
        prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

        p.add_primitive(*op, prim);
    } else {
        const auto layer_type_name = layer_type_name_ID(op);
        const auto layer_name = layer_type_name + ".out0";

        const auto mutable_precision1 = op->get_output_element_type(1);
        const auto output_shape1 = op->get_output_shape(1);
        const cldnn::layout mutable_layout1{cldnn::element_type_to_data_type(mutable_precision1),
                                            cldnn::format::get_default_format(output_shape1.size()),
                                            tensor_from_dims(output_shape1)};
        cldnn::memory::ptr shared_memory1{p.get_engine().allocate_memory(mutable_layout1)};

        const auto mutable_id_w1 = layer_type_name + "_md_write.1";
        const cldnn::mutable_data mutable_prim_w{mutable_id_w1, shared_memory1};
        p.add_primitive(*op, mutable_prim_w);
        inputs.push_back(cldnn::input_info(mutable_id_w1));

        const auto mutable_precision2 = op->get_output_element_type(2);
        const auto output_shape2 = op->get_output_shape(2);
        const cldnn::layout mutable_layout2{cldnn::element_type_to_data_type(mutable_precision2),
                                            cldnn::format::get_default_format(output_shape2.size()),
                                            tensor_from_dims(output_shape2)};
        cldnn::memory::ptr shared_memory2{p.get_engine().allocate_memory(mutable_layout2)};

        const auto mutable_id_w2 = layer_type_name + "_md_write.2";
        const cldnn::mutable_data mutable_prim_w2{mutable_id_w2, shared_memory2};
        p.add_primitive(*op, mutable_prim_w2);
        inputs.push_back(cldnn::input_info(mutable_id_w2));

        const auto expectedPrimInputCount = 4 + 2; // 4 operation inputs plus 2 input-outputs
        if (inputs.size() != expectedPrimInputCount) {
            OPENVINO_THROW("experimental_detectron_detection_output primitive requires 6 inputs");
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
                                                                attrs.deltas_weights};

        p.add_primitive(*op, prim);

        const auto mutable_id_r1 = layer_type_name + ".out1";
        const cldnn::mutable_data mutable_prim_r1{mutable_id_r1, {cldnn::input_info(layer_name)}, shared_memory1};
        p.add_primitive(*op, mutable_prim_r1);

        const auto mutable_id_r2 = layer_type_name + ".out2";
        const cldnn::mutable_data mutable_prim_r2{mutable_id_r2, {cldnn::input_info(layer_name)}, shared_memory2};
        p.add_primitive(*op, mutable_prim_r2);
    }
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronDetectionOutput);

}  // namespace ov::intel_gpu
