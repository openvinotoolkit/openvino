// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/experimental_detectron_roi_feature.hpp"

#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"

namespace ov {
namespace intel_gpu {

static void CreateExperimentalDetectronROIFeatureExtractorOp(ProgramBuilder& p,
                                                             const std::shared_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>& op) {
    auto inputs = p.GetInputInfo(op);

    const ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& operation_attributes = op->get_attrs();

    cldnn::experimental_detectron_roi_feature_extractor prim(layer_type_name_ID(op),
                                                             inputs,
                                                             operation_attributes.output_size,
                                                             operation_attributes.pyramid_scales,
                                                             operation_attributes.sampling_ratio,
                                                             operation_attributes.aligned);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronROIFeatureExtractor);

}  // namespace intel_gpu
}  // namespace ov
