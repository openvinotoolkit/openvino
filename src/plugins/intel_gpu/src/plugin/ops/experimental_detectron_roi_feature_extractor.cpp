// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/experimental_detectron_roi_feature.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"

namespace ov {
namespace intel_gpu {

static void CreateExperimentalDetectronROIFeatureExtractorOp(Program& p, const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor>& op) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op) + ".0";

    cldnn::layout mutableLayout = cldnn::layout(
        DataTypeFromPrecision(op->get_output_element_type(1)),
        DefaultFormatForDims(op->get_output_shape(1).size()),
        tensor_from_dims(op->get_output_shape(1)));

    cldnn::memory::ptr shared_memory {p.GetEngine().allocate_memory(mutableLayout)};

    cldnn::primitive_id experimental_detectron_mutable_id_w = layer_type_name_ID(op) + "_md_write";
    cldnn::mutable_data experimenta_detectron_mutable_prim(experimental_detectron_mutable_id_w,
                                                           shared_memory,
                                                           op->get_friendly_name());
    p.primitiveIDs[experimental_detectron_mutable_id_w] = experimental_detectron_mutable_id_w;
    p.AddPrimitive(experimenta_detectron_mutable_prim);
    inputPrimitives.push_back(experimental_detectron_mutable_id_w);

    const ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& operation_attributes = op->get_attrs();

    cldnn::experimental_detectron_roi_feature_extractor experimentalDetectronPrim(layerName,
                                                                                  inputPrimitives,
                                                                                  operation_attributes.output_size,
                                                                                  operation_attributes.pyramid_scales,
                                                                                  operation_attributes.sampling_ratio,
                                                                                  operation_attributes.aligned);
    p.AddPrimitive(experimentalDetectronPrim);

    cldnn::primitive_id experimental_detectron_mutable_id_r = layer_type_name_ID(op) + ".1";
    cldnn::mutable_data experimental_detectron_mutable_prim_r(experimental_detectron_mutable_id_r,
                                                              {layerName},
                                                              shared_memory,
                                                              op->get_friendly_name());
    p.primitiveIDs[experimental_detectron_mutable_id_r] = experimental_detectron_mutable_id_r;
    p.AddPrimitive(experimental_detectron_mutable_prim_r);

    p.AddPrimitiveToProfiler(layerName, op);
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronROIFeatureExtractor);

}  // namespace intel_gpu
}  // namespace ov
