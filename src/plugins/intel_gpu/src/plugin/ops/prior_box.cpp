// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/prior_box.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/primitives/prior_box.hpp"
#include "ngraph/op/prior_box_clustered.hpp"

namespace ov {
namespace intel_gpu {

static void CreatePriorBoxClusteredOp(Program& p, const std::shared_ptr<ngraph::op::v0::PriorBoxClustered>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();
    cldnn::prior_box_attributes pr_attrs;

    pr_attrs.widths = attrs.widths;
    pr_attrs.heights = attrs.heights;
    pr_attrs.step_widths = attrs.step_widths;
    pr_attrs.step_heights = attrs.step_heights;
    pr_attrs.variance = attrs.variances;
    pr_attrs.clip = attrs.clip;
    pr_attrs.offset = attrs.offset;
    pr_attrs.step = attrs.step;

    pr_attrs.min_size = {};
    pr_attrs.max_size = {};
    pr_attrs.aspect_ratio = {};
    pr_attrs.fixed_size = {};
    pr_attrs.fixed_ratio = {};
    pr_attrs.density = {};
    pr_attrs.flip = false;
    pr_attrs.clip = attrs.clip;
    pr_attrs.scale_all_sizes = false;
    pr_attrs.offset = attrs.offset;
    pr_attrs.step = attrs.step;
    pr_attrs.min_max_aspect_ratios_order = false;

    auto img_pshape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(img_pshape.is_static(), "Dynamic shapes are not supported for PriorBox operation yet");

    auto output_size_dims = op->get_input_shape(0);
    auto img_dims = op->get_input_shape(1);

    auto width = output_size_dims[0];
    auto height = output_size_dims[1];
    auto image_width = img_dims[0];
    auto image_height = img_dims[1];

    auto priorBoxPrim = cldnn::prior_box(layerName,
                                         inputPrimitives,
                                         height,
                                         width,
                                         image_height,
                                         image_width,
                                         pr_attrs,
                                         DataTypeFromPrecision(op->get_output_element_type(0)));

    p.AddPrimitive(priorBoxPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreatePriorBoxOp(Program& p, const std::shared_ptr<ngraph::op::v0::PriorBox>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();
    cldnn::prior_box_attributes pr_attrs;

    pr_attrs.widths = {};
    pr_attrs.heights = {};
    pr_attrs.step_widths = false;
    pr_attrs.step_heights = false;
    pr_attrs.min_size = attrs.min_size;
    pr_attrs.max_size = attrs.max_size;
    pr_attrs.aspect_ratio = attrs.aspect_ratio;
    pr_attrs.variance = attrs.variance;
    pr_attrs.fixed_size = attrs.fixed_size;
    pr_attrs.fixed_ratio = attrs.fixed_ratio;
    pr_attrs.density = attrs.density;
    pr_attrs.flip = attrs.flip;
    pr_attrs.clip = attrs.clip;
    pr_attrs.scale_all_sizes = attrs.scale_all_sizes;
    pr_attrs.offset = attrs.offset;
    pr_attrs.step = attrs.step;
    pr_attrs.min_max_aspect_ratios_order = false;

    auto output_size_dims = op->get_input_shape(0);
    auto img_dims = op->get_input_shape(1);

    auto img_pshape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(img_pshape.is_static(), "Dynamic shapes are not supported for PriorBox operation yet");

    auto width = output_size_dims[0];
    auto height = output_size_dims[1];
    auto image_width = img_dims[0];
    auto image_height = img_dims[1];

    auto priorBoxPrim = cldnn::prior_box(layerName,
                                         inputPrimitives,
                                         height,
                                         width,
                                         image_height,
                                         image_width,
                                         pr_attrs,
                                         DataTypeFromPrecision(op->get_output_element_type(0)));

    p.AddPrimitive(priorBoxPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreatePriorBoxOp(Program& p, const std::shared_ptr<ngraph::op::v8::PriorBox>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();
    cldnn::prior_box_attributes pr_attrs;

    pr_attrs.widths = {};
    pr_attrs.heights = {};
    pr_attrs.step_widths = false;
    pr_attrs.step_heights = false;
    pr_attrs.min_size = attrs.min_size;
    pr_attrs.max_size = attrs.max_size;
    pr_attrs.aspect_ratio = attrs.aspect_ratio;
    pr_attrs.variance = attrs.variance;
    pr_attrs.fixed_size = attrs.fixed_size;
    pr_attrs.fixed_ratio = attrs.fixed_ratio;
    pr_attrs.density = attrs.density;
    pr_attrs.flip = attrs.flip;
    pr_attrs.clip = attrs.clip;
    pr_attrs.scale_all_sizes = attrs.scale_all_sizes;
    pr_attrs.offset = attrs.offset;
    pr_attrs.step = attrs.step;
    pr_attrs.min_max_aspect_ratios_order = attrs.min_max_aspect_ratios_order;

    auto img_pshape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(img_pshape.is_static(), "Dynamic shapes are not supported for PriorBox operation yet");

    auto output_size_dims = op->get_input_shape(0);
    auto img_dims = op->get_input_shape(1);

    auto width = output_size_dims[0];
    auto height = output_size_dims[1];
    auto image_width = img_dims[0];
    auto image_height = img_dims[1];

    auto priorBoxPrim = cldnn::prior_box(layerName,
                                         inputPrimitives,
                                         height,
                                         width,
                                         image_height,
                                         image_width,
                                         pr_attrs,
                                         DataTypeFromPrecision(op->get_output_element_type(0)));

    p.AddPrimitive(priorBoxPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, PriorBoxClustered);
REGISTER_FACTORY_IMPL(v0, PriorBox);
REGISTER_FACTORY_IMPL(v8, PriorBox);

}  // namespace intel_gpu
}  // namespace ov
