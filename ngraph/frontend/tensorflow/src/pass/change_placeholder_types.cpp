// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "change_placeholder_types.hpp"

#include <memory>
#include <ngraph/graph_util.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <numeric>
#include <openvino/opsets/opset8.hpp>

#include "transformations/rt_info/old_api_map_attribute.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;

namespace {
bool is_node_casts_to_float_or_shapeof(const ov::Node* node) {
    if (dynamic_cast<const ngraph::opset8::ShapeOf*>(node)) {
        return true;
    }
    auto convert = dynamic_cast<const ngraph::opset8::Convert*>(node);
    if (convert && convert->get_destination_type() == element::f32) {
        return true;
    }

    return false;
}
}  // namespace

bool ov::frontend::tf::pass::ChangePlaceholderTypes::run_on_function(shared_ptr<Function> f) {
    for (auto param : f->get_parameters()) {
        ov::element::Type legacy_type = ov::element::undefined;
        bool all_castable_or_shapeof = true;
        for (const auto& target_input : param->get_output_target_inputs(0)) {
            all_castable_or_shapeof &= is_node_casts_to_float_or_shapeof(target_input.get_node());
        }
        if (all_castable_or_shapeof) {
            legacy_type = ov::element::f32;
        }

        if (param->get_element_type() == ngraph::element::i64) {
            legacy_type = ov::element::i32;
        }

        if (param->get_element_type() == ngraph::element::u8) {
            legacy_type = ov::element::f32;
        }

        // add or update OldApiMap only if legacy_type is defined
        if (legacy_type != ov::element::undefined) {
            // in case of existing OldApiMap we need to copy order (permutation vector)
            // into new OldApiMap with required legacy type
            std::vector<uint64_t> new_order = {};
            if (has_old_api_map(param)) {
                new_order = get_old_api_map(param).get().get_order();
            } else if (param->get_partial_shape().rank().is_static()) {
                auto rank = param->get_partial_shape().rank().get_length();
                new_order.resize(rank);
                std::iota(new_order.begin(), new_order.end(), 0);
            }

            auto old_api_map = std::make_shared<ov::OldApiMap>(ov::OldApiMapAttr(new_order, legacy_type));
            set_old_api_map(std::dynamic_pointer_cast<ov::Node>(param), old_api_map->get());
        }
    }

    return true;
}
