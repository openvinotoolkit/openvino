// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_detection_output.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/experimental_detectron_detection_output.hpp"

namespace ov {
namespace intel_gpu {

static void CreateExperimentalDetectronDetectionOutputOp(
    ProgramBuilder& p,
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>& op) {
    validate_inputs_count(op, {4});

    if (op->get_output_size() != 3) {
        OPENVINO_THROW("ExperimentalDetectronDetectionOutput requires 3 outputs");
    }

    auto inputs = p.GetInputInfo(op);

    const auto& attrs = op->get_attrs();

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
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronDetectionOutput);

}  // namespace intel_gpu
}  // namespace ov
