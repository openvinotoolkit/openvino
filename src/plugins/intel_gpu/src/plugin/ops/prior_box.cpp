// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box.hpp"
#include "openvino/op/prior_box_clustered.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/prior_box.hpp"

namespace ov::intel_gpu {

static void CreatePriorBoxClusteredOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::PriorBoxClustered>& op) {
    OPENVINO_ASSERT(false, "[GPU] PriorBoxClustered op is not supported in GPU plugin yet.");
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();

    std::vector<float> width = attrs.widths;
    std::vector<float> height = attrs.heights;
    std::vector<float> variance = attrs.variances;
    float offset = attrs.offset;
    bool clip = attrs.clip;

    auto input_pshape = op->get_input_partial_shape(0);
    auto img_pshape = op->get_input_partial_shape(1);
    auto output_pshape = op->get_output_partial_shape(0);

    OPENVINO_ASSERT(input_pshape.is_static() && img_pshape.is_static(), "Dynamic shapes are not supported for PriorBoxClustered operation yet");

    if (!output_pshape.is_dynamic()) {
        auto input_shape = input_pshape.to_shape();
        auto img_shape = img_pshape.to_shape();

        int img_w = static_cast<int>(img_shape.back());
        int img_h = static_cast<int>(img_shape.at(img_shape.size() - 2));
        cldnn::tensor img_size = (cldnn::tensor) cldnn::spatial(TensorValue(img_w), TensorValue(img_h));

        auto step_w = attrs.step_widths;
        auto step_h = attrs.step_heights;
        if (std::abs(attrs.step_heights - attrs.step_widths) < 1e-5) {
            step_w = attrs.step_widths;
            step_h = attrs.step_widths;
        }

        if (step_w == 0.0f && step_h == 0.0f) {
            step_w = static_cast<float>(img_w) / input_shape.back();
            step_h = static_cast<float>(img_h) / input_shape.at(img_shape.size() - 2);
        }

        auto priorBoxPrim = cldnn::prior_box(layerName,
                                             inputs,
                                             img_size,
                                             clip,
                                             variance,
                                             step_w,
                                             step_h,
                                             offset,
                                             width,
                                             height,
                                             cldnn::element_type_to_data_type(op->get_output_element_type(0)));

        p.add_primitive(*op, priorBoxPrim);
    } else {
        auto step_w = attrs.step_widths;
        auto step_h = attrs.step_heights;
        cldnn::tensor img_size{};
        auto priorBoxPrim = cldnn::prior_box(layerName,
                                             inputs,
                                             img_size,
                                             clip,
                                             variance,
                                             step_w,
                                             step_h,
                                             offset,
                                             width,
                                             height,
                                             cldnn::element_type_to_data_type(op->get_output_element_type(0)));

        p.add_primitive(*op, priorBoxPrim);
    }
}

static void CreatePriorBoxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::PriorBox>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
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
    auto step = attrs.step;

    auto output_pshape = op->get_output_partial_shape(0);
    auto img_pshape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(img_pshape.is_static(), "Dynamic shapes are not supported for PriorBox operation yet");

    if (!output_pshape.is_dynamic()) {
        const auto output_size_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(0));
        const auto image_size_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));

        // output_size should be constant to be static output shape
        OPENVINO_ASSERT(output_size_constant,
                        "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        const auto output_size = output_size_constant->cast_vector<int64_t>();
        const auto width = output_size[0];
        const auto height = output_size[1];
        const cldnn::tensor output_size_tensor{cldnn::spatial(width, height)};

        cldnn::tensor img_size_tensor{};
        // When image size is constant, set the value for primitive construction. Others don't have to set it. It will be determined in execute_impl time.
        if (image_size_constant) {
            const auto image_size = image_size_constant->cast_vector<int64_t>();
            const auto image_width = image_size[0];
            const auto image_height = image_size[1];
            img_size_tensor = (cldnn::tensor) cldnn::spatial(image_width, image_height);
        }

        auto priorBoxPrim = cldnn::prior_box(layerName,
                                             inputs,
                                             output_size_tensor,
                                             img_size_tensor,
                                             min_size,
                                             max_size,
                                             aspect_ratio,
                                             flip,
                                             clip,
                                             variance,
                                             step,
                                             offset,
                                             scale_all_sizes,
                                             fixed_ratio,
                                             fixed_size,
                                             density);

        p.add_primitive(*op, priorBoxPrim);
    } else {
        cldnn::tensor output_size{};
        cldnn::tensor img_size{};
        auto priorBoxPrim = cldnn::prior_box(layerName,
                                             inputs,
                                             output_size,
                                             img_size,
                                             min_size,
                                             max_size,
                                             aspect_ratio,
                                             flip,
                                             clip,
                                             variance,
                                             step,
                                             offset,
                                             scale_all_sizes,
                                             fixed_ratio,
                                             fixed_size,
                                             density);

        p.add_primitive(*op, priorBoxPrim);
    }
}

static void CreatePriorBoxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::PriorBox>& op) {
    validate_inputs_count(op, {2});
    const auto inputs = p.GetInputInfo(op);
    std::string layer_name = layer_type_name_ID(op);

    const auto& attrs = op->get_attrs();
    auto output_pshape = op->get_output_partial_shape(0);

    if (!output_pshape.is_dynamic()) {
        const auto output_size_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(0));
        const auto image_size_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));

        // output_size should be constant to be static output shape
        OPENVINO_ASSERT(output_size_constant,
                        "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        const auto output_size = output_size_constant->cast_vector<int64_t>();
        const auto width = output_size[0];
        const auto height = output_size[1];
        const cldnn::tensor output_size_tensor{cldnn::spatial(width, height)};

        cldnn::tensor img_size_tensor{};
        // When image size is constant, set the value for primitive construction. Others don't have to set it. It will be determined in execute_impl time.
        if (image_size_constant) {
            const auto image_size = image_size_constant->cast_vector<int64_t>();
            const auto image_width = image_size[0];
            const auto image_height = image_size[1];
            img_size_tensor = (cldnn::tensor) cldnn::spatial(image_width, image_height);
        }

        const cldnn::prior_box prior_box{layer_name,
                                         inputs,
                                         output_size_tensor,
                                         img_size_tensor,
                                         attrs.min_size,
                                         attrs.max_size,
                                         attrs.aspect_ratio,
                                         attrs.flip,
                                         attrs.clip,
                                         attrs.variance,
                                         attrs.step,
                                         attrs.offset,
                                         attrs.scale_all_sizes,
                                         attrs.fixed_ratio,
                                         attrs.fixed_size,
                                         attrs.density,
                                         true,
                                         attrs.min_max_aspect_ratios_order};

        p.add_primitive(*op, prior_box);
    } else {
        cldnn::tensor output_size{};
        cldnn::tensor img_size{};

        const cldnn::prior_box prior_box{layer_name,
                                         inputs,
                                         output_size,
                                         img_size,
                                         attrs.min_size,
                                         attrs.max_size,
                                         attrs.aspect_ratio,
                                         attrs.flip,
                                         attrs.clip,
                                         attrs.variance,
                                         attrs.step,
                                         attrs.offset,
                                         attrs.scale_all_sizes,
                                         attrs.fixed_ratio,
                                         attrs.fixed_size,
                                         attrs.density,
                                         true,
                                         attrs.min_max_aspect_ratios_order};

        p.add_primitive(*op, prior_box);
    }
}

REGISTER_FACTORY_IMPL(v0, PriorBoxClustered);
REGISTER_FACTORY_IMPL(v0, PriorBox);
REGISTER_FACTORY_IMPL(v8, PriorBox);

}  // namespace ov::intel_gpu
