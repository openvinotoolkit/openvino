// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/experimental_detectron_roi_feature.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"

namespace ov {
namespace intel_gpu {

static void CreateExperimentalDetectronROIFeatureExtractorOp(ProgramBuilder& p,
                                                             const std::shared_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>& op) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op) + ".out0";

    cldnn::layout mutableLayout = cldnn::layout(
        cldnn::element_type_to_data_type(op->get_output_element_type(1)),
        cldnn::format::get_default_format(op->get_output_shape(1).size()),
        tensor_from_dims(op->get_output_shape(1)));

    cldnn::memory::ptr shared_memory {p.get_engine().allocate_memory(mutableLayout)};

    cldnn::primitive_id experimental_detectron_mutable_id_w = layer_type_name_ID(op) + "_md_write";
    cldnn::mutable_data experimenta_detectron_mutable_prim(experimental_detectron_mutable_id_w,
                                                           shared_memory);
    p.add_primitive(*op, experimenta_detectron_mutable_prim);
    inputs.push_back(cldnn::input_info(experimental_detectron_mutable_id_w));

    const ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& operation_attributes = op->get_attrs();

    cldnn::experimental_detectron_roi_feature_extractor experimentalDetectronPrim(layerName,
                                                                                  inputs,
                                                                                  operation_attributes.output_size,
                                                                                  operation_attributes.pyramid_scales,
                                                                                  operation_attributes.sampling_ratio,
                                                                                  operation_attributes.aligned);
    p.add_primitive(*op, experimentalDetectronPrim);

    cldnn::primitive_id experimental_detectron_mutable_id_r = layer_type_name_ID(op) + ".out1";
    cldnn::mutable_data experimental_detectron_mutable_prim_r(experimental_detectron_mutable_id_r,
                                                              {cldnn::input_info(layerName)},
                                                              shared_memory);
    p.add_primitive(*op, experimental_detectron_mutable_prim_r);
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronROIFeatureExtractor);

}  // namespace intel_gpu
}  // namespace ov
