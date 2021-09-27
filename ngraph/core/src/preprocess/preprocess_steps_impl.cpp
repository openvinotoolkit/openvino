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
    OPENVINO_ASSERT(layout::has_channels(context.layout()), "Channels dimension is not specified in layout");
    auto channels_index = layout::channels(context.layout());
    auto node_shape = node->get_output_partial_shape(0);
    auto node_rank = node->get_output_partial_shape(0).rank();
    OPENVINO_ASSERT(node_rank.is_static(), "Mean/scale vector operation is not supported for fully dynamic shape");
    std::vector<std::size_t> v(node_rank.get_length(), 1);
    if (channels_index < 0) {
        // E.g. channels_index = -1 means last dimension
        channels_index = node_rank.get_length() + channels_index;
    }
    OPENVINO_ASSERT(node_rank.get_length() > channels_index, "Channels dimension is out of bounds");

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

}  // namespace preprocess
}  // namespace ov
