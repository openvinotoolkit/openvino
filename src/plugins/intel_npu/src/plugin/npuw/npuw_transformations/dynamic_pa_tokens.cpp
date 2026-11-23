// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_pa_tokens.hpp"

#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

bool ov::npuw::DynamicPATokens::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<ov::PartialShape> result_shapes;
    for (const auto& result : model->get_results()) {
        result_shapes.push_back(result->get_input_partial_shape(0));
    }

    bool relaxed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()) != "PagedAttentionExtension") {
            continue;
        }
        // Inputs 0, 1 and 2 are the queries, keys and values, [tokens, heads * head_size].
        for (std::size_t i = 0; i < 3; ++i) {
            const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node->get_input_node_shared_ptr(i));
            if (!param) {
                continue;
            }
            auto shape = param->get_partial_shape();
            if (shape.rank().is_dynamic() || shape.size() == 0 || shape[0].is_dynamic()) {
                continue;
            }
            shape[0] = ov::Dimension::dynamic();
            param->set_partial_shape(shape);
            relaxed = true;
        }
    }
    if (!relaxed) {
        return false;
    }
    model->validate_nodes_and_infer_types();

    const auto& results = model->get_results();
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& shape = result_shapes[i];
        if (results[i]->get_input_partial_shape(0) == shape || shape.is_dynamic()) {
            continue;
        }
        const auto dims = shape.to_shape();
        auto target = ov::op::v0::Constant::create(ov::element::i64,
                                                   ov::Shape{dims.size()},
                                                   std::vector<int64_t>(dims.begin(), dims.end()));
        auto source = results[i]->input_value(0);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(source, target, false);
        reshape->output(0).get_tensor().set_names(source.get_names());
        source.get_tensor().set_names({});
        results[i]->input(0).replace_source_output(reshape);
    }
    model->validate_nodes_and_infer_types();
    return true;
}
