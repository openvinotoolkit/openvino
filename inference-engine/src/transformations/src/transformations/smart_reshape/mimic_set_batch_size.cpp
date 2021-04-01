// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/smart_reshape/mimic_set_batch_size.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MimicSetBatchSize, "MimicSetBatchSize", 0);

bool ngraph::pass::MimicSetBatchSize::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(MimicSetBatchSize);
    // extracting ratio of out to in 0-index dimension value from the folded function
    auto specialized_function = ngraph::clone_function(*f);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.run_passes(specialized_function);

    std::map<std::string, float> scale;
    for (const auto & node : specialized_function->get_ops()) {
        if (const auto & reshape = std::dynamic_pointer_cast<opset5::Reshape>(node)) {
            const auto in_pshape = reshape->get_input_partial_shape(0), out_pshape = reshape->get_output_partial_shape(0);
            if (in_pshape.rank().is_dynamic() || in_pshape.rank().get_length() <= 1 || in_pshape[0].is_dynamic() ||
                out_pshape.rank().is_dynamic() || out_pshape.rank().get_length() <= 1 || out_pshape[0].is_dynamic())
                continue;
            const auto & pattern = std::dynamic_pointer_cast<opset5::Constant>(reshape->get_input_node_shared_ptr(1));
            if (pattern && pattern->cast_vector<int64_t>()[0] > 0) {
                scale[reshape->get_friendly_name()] = static_cast<float>(out_pshape[0].get_length()) / static_cast<float>(in_pshape[0].get_length());
            }
        }
    }
    // apply transformation to original function
    bool transformed = false;
    for (auto & reshape : f->get_ops()) {
        if (!is_type<opset5::Reshape>(reshape) || !scale.count(reshape->get_friendly_name()) || reshape->get_output_partial_shape(0).rank().is_dynamic())
            continue;

        const auto & shape_of = std::make_shared<opset5::ShapeOf>(reshape->get_input_source_output(0), reshape->get_input_element_type(1));
        const auto & new_input_batch = std::make_shared<ngraph::opset5::Gather>(
                shape_of, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));

        const std::shared_ptr<Node> & new_output_batch = std::make_shared<opset5::Convert>(
                std::make_shared<opset5::Ceiling>(
                        std::make_shared<opset5::Multiply>(
                                std::make_shared<opset5::Convert>(new_input_batch, element::f32),
                                opset5::Constant::create(element::f32, {1}, {scale[reshape->get_friendly_name()]}))),
                reshape->get_input_element_type(1));

        std::vector<int64_t> non_batch_dims(reshape->get_output_partial_shape(0).rank().get_length() - 1);
        std::iota(non_batch_dims.begin(), non_batch_dims.end(), 1);
        const auto & non_batch_dims_node = std::make_shared<ngraph::opset5::Gather>(
                reshape->input_value(1),
                ngraph::opset5::Constant::create(ngraph::element::i64, {non_batch_dims.size()}, non_batch_dims),
                ngraph::opset5::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));
        auto new_reshape_pattern = std::make_shared<opset5::Concat>(OutputVector{new_output_batch, non_batch_dims_node}, 0);
        reshape->input(1).replace_source_output(new_reshape_pattern->output(0));
        transformed = true;
    }
    return transformed;
}
