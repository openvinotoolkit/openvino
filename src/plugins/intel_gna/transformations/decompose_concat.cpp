// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/decompose_concat.hpp"

#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>
#include "backend/gna_limitations.hpp"


using namespace ov::intel_gna::pass;
using namespace ngraph;

struct ConcatData {
    size_t num_inputs;
    size_t split_axis = 0;
    size_t leading_shape_product = 1;
    int64_t axis;
    OutputVector concat_parents;
};

static bool GetVerifiedConcatData(const std::shared_ptr<opset9::Concat> concat, ConcatData& concat_data) {
    std::vector<ov::Shape> input_shape;
    concat_data.num_inputs = concat->inputs().size();

    for (size_t i = 0; i < concat_data.num_inputs; i++) {
        concat_data.concat_parents.push_back(concat->input_value(i));
        input_shape.push_back(concat->input_value(i).get_shape());
    }

    concat_data.axis = concat->get_axis();
    int32_t non_one_axis_count = 0;

    for (int64_t i = 0; i < concat_data.axis; i++) {
        if (input_shape[0][i] != 1) {
            concat_data.leading_shape_product *= input_shape[0][i];
            concat_data.split_axis = i;
            non_one_axis_count++;
        }
    }
    // Simple Concats are GNA-compatible already
    if (concat_data.leading_shape_product == 1 ||
        // Difficult cases not yet implemented
        non_one_axis_count > 1) {
        return false;
    }

    return true;
}

static void Decompose(const std::shared_ptr<opset9::Concat> concat, const ConcatData& concat_data) {
    OutputVector splits;
    OutputVector chunks;

    for (size_t i = 0; i < concat_data.num_inputs; i++) {
        const auto axis_node = opset9::Constant::create(element::i64, Shape{}, {concat_data.split_axis});
        const auto split = std::make_shared<opset9::Split>(concat_data.concat_parents[i], axis_node, concat_data.leading_shape_product);
        splits.push_back(split);
    }

    for (size_t c = 0; c < concat_data.leading_shape_product; c++) {
        OutputVector sub_chunks;
        for (size_t i = 0; i < concat_data.num_inputs; i++) {
            sub_chunks.push_back(splits[i].get_node()->output(c));
        }
        auto new_sub_concat = std::make_shared<opset9::Concat>(sub_chunks, concat_data.axis);
        chunks.push_back(new_sub_concat->output(0));
    }

    auto new_concat = std::make_shared<opset9::Concat>(chunks, concat_data.split_axis);
    replace_node(concat, new_concat);
    new_concat->set_friendly_name(concat->get_friendly_name());
}

static bool Convert(std::shared_ptr<Node> concat_node) {
    const auto concat = std::dynamic_pointer_cast<opset9::Concat>(concat_node);
    ConcatData concat_data = {};

    if (!GetVerifiedConcatData(concat, concat_data))
        return false;

    Decompose(concat, concat_data);

    return true;
}

DecomposeConcat::DecomposeConcat() {
    MATCHER_SCOPE(DecomposeConcat);

    auto concat = pattern::wrap_type<opset9::Concat>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(concat).get_node_shared_ptr());
    };

    auto m = std::make_shared<pattern::Matcher>(concat, matcher_name);
    this->register_matcher(m, callback);
}
