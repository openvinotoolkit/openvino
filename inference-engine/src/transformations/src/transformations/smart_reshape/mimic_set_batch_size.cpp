// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/smart_reshape/mimic_set_batch_size.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MimicSetBatchSize, "MimicSetBatchSize", 0);

ngraph::pass::MimicSetBatchSize::MimicSetBatchSize() {
    auto reshape_label = ngraph::pattern::wrap_type<opset5::Reshape>({pattern::any_input(), ngraph::pattern::wrap_type<opset5::Constant>()},
         [](const Output<Node> &output) {
             const auto &input_pshape = output.get_node_shared_ptr()->get_input_partial_shape(0);
             const auto &output_rank = output.get_partial_shape().rank();
             return input_pshape.rank().is_static() && input_pshape.rank().get_length() > 1 && input_pshape[0].is_static() &&
                    output_rank.is_static() && output_rank.get_length() > 1;
         });

    matcher_pass_callback callback = [=](pattern::Matcher &m) -> bool {
        const auto & reshape = m.get_match_root();
        auto pattern = std::dynamic_pointer_cast<opset5::Constant>(reshape->get_input_node_shared_ptr(1));
        if (!pattern)
            return false;

        const auto & pattern_vector = pattern->cast_vector<int64_t>();
        if (pattern_vector.empty() || pattern_vector[0] < 1)
            return false;

        // mimicking old setBatchSize style (copied):
        // float diff = static_cast<float>(dims.at(0)) / static_cast<float>(originalBatchSize);
        // dims.at(0) = static_cast<size_t>(std::ceil(size * diff));

        const auto & old_input_batch = static_cast<float>(reshape->get_input_partial_shape(0)[0].get_length());
        const auto & old_output_batch = static_cast<float>(pattern_vector[0]);

        const auto & scale = old_output_batch / old_input_batch;

        const auto & shape_of = std::make_shared<opset5::ShapeOf>(reshape->get_input_source_output(0), pattern->get_element_type());
        const auto & new_input_batch = std::make_shared<ngraph::opset5::Gather>(
                shape_of, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));

        const std::shared_ptr<Node> & new_output_batch = std::make_shared<opset5::Convert>(
                std::make_shared<opset5::Ceiling>(
                        std::make_shared<opset5::Multiply>(
                                std::make_shared<opset5::Convert>(new_input_batch, element::f32),
                                opset5::Constant::create(element::f32, {1}, {scale}))),
                pattern->get_element_type());

        auto new_reshape_pattern = new_output_batch;
        const auto rank = pattern_vector.size();
        if (rank > 1) {
            std::vector<int64_t> non_batch_dims(rank - 1);
            std::iota(non_batch_dims.begin(), non_batch_dims.end(), 1);
            const auto & non_batch_dims_node = std::make_shared<ngraph::opset5::Gather>(
                    pattern,
                    ngraph::opset5::Constant::create(ngraph::element::i64, {non_batch_dims.size()}, non_batch_dims),
                    ngraph::opset5::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));
            new_reshape_pattern = std::make_shared<opset5::Concat>(OutputVector{new_reshape_pattern, non_batch_dims_node}, 0);
        }
        reshape->input(1).replace_source_output(new_reshape_pattern->output(0));
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_label, "MimicSetBatchSize");
    register_matcher(m, callback);
}
