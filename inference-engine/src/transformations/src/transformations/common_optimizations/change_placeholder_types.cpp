// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/change_placeholder_types.hpp"

#include <memory>
#include <numeric>

#include "openvino/core/node.hpp"
#include "openvino/opsets/opset8.hpp"
#include "transformations/rt_info/old_api_map_attribute.hpp"

using namespace std;
using namespace ov;

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

bool ov::pass::ChangePlaceholderTypes::run_on_function(shared_ptr<Function> f) {
  for (const auto& param : f->get_parameters()) {
    element::Type legacy_type = element::undefined;
    bool all_castable_or_shapeof = true;
    for (const auto& target_input : param->get_output_target_inputs(0)) {
      all_castable_or_shapeof &=
          is_node_casts_to_float_or_shapeof(target_input.get_node());
    }
    if (all_castable_or_shapeof) {
      legacy_type = element::f32;
    }

    if (param->get_element_type() == element::i64) {
      legacy_type = element::i32;
    } else if (param->get_element_type() == element::u8) {
      legacy_type = element::f32;
    }

    // add or update OldApiMap only if legacy_type is defined
    if (legacy_type != element::undefined) {
      // in case of existing OldApiMap we need to copy order (permutation
      // vector) into new OldApiMap with required legacy type
      std::vector<uint64_t> new_order = {};
      if (has_old_api_map(param)) {
        new_order = get_old_api_map(param).get().get_order();
      } else if (param->get_partial_shape().rank().is_static()) {
        auto rank = param->get_partial_shape().rank().get_length();
        new_order.resize(rank);
        std::iota(new_order.begin(), new_order.end(), 0);
      }
      set_old_api_map(param, OldApiMapAttr(new_order, legacy_type));
    }
  }

  return true;
}
