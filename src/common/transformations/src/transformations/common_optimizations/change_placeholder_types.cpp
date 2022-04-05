// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/change_placeholder_types.hpp"

#include <algorithm>
#include <memory>
#include <numeric>

#include "ngraph/node.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"

using namespace std;
using namespace ngraph;

namespace {
bool is_node_casts_to_float_or_shapeof(const Node* node) {
    if (dynamic_cast<const opset8::ShapeOf*>(node)) {
        return true;
    }
    auto convert = dynamic_cast<const opset8::Convert*>(node);
    if (convert && convert->get_destination_type() == element::f32) {
        return true;
    }

    return false;
}
}  // namespace

bool ov::pass::ChangePlaceholderTypes::run_on_model(const shared_ptr<ov::Model>& f) {
    for (auto& param : f->get_parameters()) {
        // do not set legacy type if an user defines own type
        auto& param_name = param->get_friendly_name();
        if (std::find(m_params_with_custom_types.begin(), m_params_with_custom_types.end(), param_name) !=
            m_params_with_custom_types.end())
            continue;

        element::Type legacy_type = element::undefined;
        bool all_castable_or_shapeof = true;
        for (const auto& target_input : param->get_output_target_inputs(0)) {
            all_castable_or_shapeof &= is_node_casts_to_float_or_shapeof(target_input.get_node());
        }
        if (all_castable_or_shapeof) {
            legacy_type = element::f32;
        }

        if (param->get_element_type() == element::i64) {
            legacy_type = element::i32;
        } else if (param->get_element_type() == element::u8) {
            legacy_type = element::f32;
        }

        // set OldApiMapElementType only if legacy_type is defined
        if (legacy_type != element::undefined) {
            set_old_api_map_element_type(param, ov::OldApiMapElementType(legacy_type));
        }
    }
    return true;
}
