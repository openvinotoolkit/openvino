// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocess_steps_impl.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace preprocess {

static Shape construct_mean_scale_shape(const std::shared_ptr<Node>& node,
                                        size_t values_size,
                                        const PreprocessingContext& context) {
    // TODO: support also Mean/Scale image case
    auto node_shape = node->get_output_partial_shape(0);
    auto node_rank = node_shape.rank();
    auto channels_index = get_and_check_channels_idx(context.layout(), node_shape);
    std::vector<std::size_t> v(node_rank.get_length(), 1);

    OPENVINO_ASSERT(node_shape[channels_index].is_dynamic() || node_shape[channels_index] == values_size,
                    "Number of channels and mean/values size mismatch: Channels = ",
                    node_shape[channels_index].get_length(),
                    ", mean/scale = ",
                    values_size);
    v[channels_index] = values_size;
    return {v};
}

void PreProcessSteps::PreProcessStepsImpl::add_scale_impl(const std::vector<float>& values) {
    m_actions.emplace_back(std::make_tuple(
        [values](const std::vector<std::shared_ptr<Node>>& nodes, PreprocessingContext& context) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't apply scale preprocessing for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply scale preprocessing for multi-plane input. Suggesting to convert current "
                            "image to RGB/BGR color format using 'convert_color'");
            Shape shape;
            if (values.size() == 1) {
                shape = Shape{1};
            } else {
                shape = construct_mean_scale_shape(nodes[0], values.size(), context);
            }
            auto constant = op::v0::Constant::create(element::f32, shape, values);
            constant->set_friendly_name(nodes[0]->get_friendly_name() + "/scale/Divide_Factor");

            auto new_op = std::make_shared<op::v1::Divide>(nodes[0], constant);
            new_op->set_friendly_name(nodes[0]->get_friendly_name() + "/scale/Divide");
            return new_op;
        },
        false));
}

void PreProcessSteps::PreProcessStepsImpl::add_mean_impl(const std::vector<float>& values) {
    m_actions.emplace_back(std::make_tuple(
        [values](const std::vector<std::shared_ptr<Node>>& nodes, PreprocessingContext& context) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't apply mean preprocessing for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply scale preprocessing for multi-plane input. Suggesting to convert current "
                            "image to RGB/BGR color format using 'convert_color'");
            Shape shape;
            if (values.size() == 1) {
                shape = Shape{1};
            } else {
                shape = construct_mean_scale_shape(nodes[0], values.size(), context);
            }
            auto constant = op::v0::Constant::create(element::f32, shape, values);
            constant->set_friendly_name(nodes[0]->get_friendly_name() + "/mean/Mean_Const");

            auto new_op = std::make_shared<op::v1::Subtract>(nodes[0], constant);
            new_op->set_friendly_name(nodes[0]->get_friendly_name() + "/mean/Subtract");
            return new_op;
        },
        false));
}

void PreProcessSteps::PreProcessStepsImpl::add_convert_impl(const ov::element::Type& type) {
    m_actions.emplace_back(std::make_tuple(
        [type](const std::vector<std::shared_ptr<Node>>& nodes, PreprocessingContext&) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't set element type for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't set element type for multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'convert_color'");
            OPENVINO_ASSERT(nodes[0]->get_element_type().is_static(),
                            "Can't insert 'convert_element_type' for dynamic source tensor type.");
            auto convert = std::make_shared<op::v0::Convert>(nodes[0], type);
            convert->set_friendly_name(nodes[0]->get_friendly_name() + "/convert_element_type");
            return convert;
        },
        true));
}

void PreProcessSteps::PreProcessStepsImpl::add_resize_impl(ResizeAlgorithm alg, int dst_height, int dst_width) {
    using InterpolateMode = op::v4::Interpolate::InterpolateMode;
    m_actions.emplace_back(std::make_tuple(
        [alg, dst_width, dst_height](const std::vector<std::shared_ptr<Node>>& nodes, PreprocessingContext& ctxt) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't add resize for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't resize multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'PreProcessSteps::convert_color'");
            auto to_mode = [](ResizeAlgorithm alg) -> InterpolateMode {
                switch (alg) {
                case ResizeAlgorithm::RESIZE_NEAREST:
                    return InterpolateMode::NEAREST;
                case ResizeAlgorithm::RESIZE_CUBIC:
                    return InterpolateMode::CUBIC;
                case ResizeAlgorithm::RESIZE_LINEAR:
                default:
                    return InterpolateMode::LINEAR;
                }
            };
            auto node = nodes.front();
            auto layout = ctxt.layout();
            OPENVINO_ASSERT(ov::layout::has_height(layout) && ov::layout::has_width(layout),
                            "Can't add resize for layout without W/H specified. Use 'set_layout' API to define layout "
                            "of image data, like `NCHW`");
            auto node_rank = node->get_output_partial_shape(0).rank();
            OPENVINO_ASSERT(node_rank.is_static(), "Resize operation is not supported for fully dynamic shape");

            auto height_idx = static_cast<int64_t>(get_and_check_height_idx(layout, node->get_output_partial_shape(0)));
            auto width_idx = static_cast<int64_t>(get_and_check_width_idx(layout, node->get_output_partial_shape(0)));
            if (dst_height < 0 || dst_width < 0) {
                OPENVINO_ASSERT(ctxt.network_shape().rank().is_static(),
                                "Resize is not fully specified while target network shape is dynamic");
            }
            int new_image_width = dst_width < 0 ? static_cast<int>(ctxt.get_network_width_for_resize()) : dst_width;
            int new_image_height = dst_height < 0 ? static_cast<int>(ctxt.get_network_height_for_resize()) : dst_height;

            auto target_spatial_shape =
                op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {new_image_height, new_image_width});
            auto scales = op::v0::Constant::create<float>(element::f32, Shape{2}, {1, 1});
            // In future consider replacing this to set of new OV operations like `getDimByName(node, "H")`
            // This is to allow specifying layout on 'evaluation' stage
            auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {height_idx, width_idx});

            op::v4::Interpolate::InterpolateAttrs attrs(to_mode(alg),
                                                        op::v4::Interpolate::ShapeCalcMode::SIZES,
                                                        {0, 0},
                                                        {0, 0});

            auto interp = std::make_shared<op::v4::Interpolate>(node, target_spatial_shape, scales, axes, attrs);
            interp->set_friendly_name(nodes[0]->get_friendly_name() + "/resize");
            return interp;
        },
        true));
}

void PreProcessSteps::PreProcessStepsImpl::add_convert_layout_impl(const Layout& layout) {
    m_actions.emplace_back(std::make_tuple(
        [layout](const std::vector<std::shared_ptr<Node>>& nodes, PreprocessingContext& context) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't convert layout for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't convert layout for multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'convert_color'");
            Layout dst_layout = layout == Layout() ? context.network_layout() : layout;
            auto permutation =
                layout::find_permutation(context.layout(), nodes[0]->get_output_partial_shape(0), dst_layout);
            auto perm_constant =
                op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
            auto transpose = std::make_shared<op::v1::Transpose>(nodes[0], perm_constant);
            transpose->set_friendly_name(nodes[0]->get_friendly_name() + "/convert_layout");
            context.layout() = dst_layout;  // Update context's current layout
            return transpose;
        },
        true));
}

}  // namespace preprocess
}  // namespace ov
