// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/prior_box.hpp"
#include "ngraph/op/prior_box_clustered.hpp"

#include "intel_gpu/primitives/prior_box.hpp"

namespace ov {
namespace intel_gpu {

static void CreatePriorBoxClusteredOp(Program& p, const std::shared_ptr<ngraph::op::v0::PriorBoxClustered>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();

    std::vector<float> width = attrs.widths;
    std::vector<float> height = attrs.heights;
    std::vector<float> variance = attrs.variances;
    float offset = attrs.offset;
    bool clip = attrs.clip;

    auto inp_dims = op->get_input_shape(0);
    auto img_dims = op->get_input_shape(1);

    int img_w = static_cast<int>(img_dims.back());
    int img_h = static_cast<int>(img_dims.at(img_dims.size() - 2));
    cldnn::tensor img_size = (cldnn::tensor) cldnn::spatial(TensorValue(img_w), TensorValue(img_h));

    auto step_w = attrs.step_widths;
    auto step_h = attrs.step_heights;
    if (std::abs(attrs.step_heights - attrs.step_widths) < 1e-5) {
        step_w = attrs.step_widths;
        step_h = attrs.step_widths;
    }

    if (step_w == 0.0f && step_h == 0.0f) {
        step_w = static_cast<float>(img_w) / inp_dims.back();
        step_h = static_cast<float>(img_h) / inp_dims.at(img_dims.size() - 2);
    }

    auto priorBoxPrim = cldnn::prior_box(layerName,
                                         inputPrimitives[0],
                                         img_size,
                                         clip,
                                         variance,
                                         step_w,
                                         step_h,
                                         offset,
                                         width,
                                         height,
                                         DataTypeFromPrecision(op->get_output_element_type(0)),
                                         op->get_friendly_name());

    p.AddPrimitive(priorBoxPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreatePriorBoxOp(Program& p, const std::shared_ptr<ngraph::op::v0::PriorBox>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();
    // params
    std::vector<float> min_size = attrs.min_size;
    std::vector<float> max_size = attrs.max_size;
    std::vector<float> aspect_ratio = attrs.aspect_ratio;
    std::vector<float> variance = attrs.variance;
    std::vector<float> fixed_size = attrs.fixed_size;
    std::vector<float> fixed_ratio = attrs.fixed_ratio;
    std::vector<float> density = attrs.density;
    bool flip = attrs.flip;
    bool clip = attrs.clip;
    bool scale_all_sizes = attrs.scale_all_sizes;
    float offset = attrs.offset;

    auto step_w = attrs.step;
    auto step_h = attrs.step;

    auto img_dims = op->get_input_shape(1);

    auto wdim = img_dims.back();
    auto hdim = img_dims.at(img_dims.size()-2);

    cldnn::tensor img_size = (cldnn::tensor) cldnn::spatial(TensorValue(wdim), TensorValue(hdim));
    auto priorBoxPrim = cldnn::prior_box(layerName,
                                         inputPrimitives[0],
                                         img_size,
                                         min_size,
                                         max_size,
                                         aspect_ratio,
                                         flip,
                                         clip,
                                         variance,
                                         step_w,
                                         step_h,
                                         offset,
                                         scale_all_sizes,
                                         fixed_ratio,
                                         fixed_size,
                                         density,
                                         op->get_friendly_name());

    p.AddPrimitive(priorBoxPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, PriorBoxClustered);
REGISTER_FACTORY_IMPL(v0, PriorBox);

}  // namespace intel_gpu
}  // namespace ov
