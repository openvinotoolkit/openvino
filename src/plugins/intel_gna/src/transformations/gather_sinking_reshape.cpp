// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_reshape.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::graph_utils;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;

int get_shapes_squeeze_shift(const Shape& shape1, const Shape& shape2) {
    const int index_1 = static_cast<int>(get_first_valuable_dim_id(shape1));
    const int index_2 = static_cast<int>(get_first_valuable_dim_id(shape2));

    if (index_1 < 0 || index_2 < 0)
        return 0;
    return index_1 - index_2;
}

}  // namespace

GatherSinkingReshapeBackward::GatherSinkingReshapeBackward() {
    MATCHER_SCOPE(GatherSinkingReshapeBackward);
    auto reshape_const_label = wrap_type<Constant>();
    auto reshape_label = wrap_type<Reshape>({any_input(), reshape_const_label}, is_reshape_unsqueeze);
    auto gather_indices_label = wrap_type<Constant>();
    auto gather_axis_label = wrap_type<Constant>();
    auto gather_label =
        wrap_type<Gather>({reshape_label, gather_indices_label, gather_axis_label}, is_gather_sinking_enabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather_indices = as_type_ptr<Constant>(pattern_to_output.at(gather_indices_label).get_node_shared_ptr());
        auto gather_axis = as_type_ptr<Constant>(pattern_to_output.at(gather_axis_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());
        auto reshape_const = as_type_ptr<Constant>(pattern_to_output.at(reshape_const_label).get_node_shared_ptr());
        auto reshape = as_type_ptr<Reshape>(pattern_to_output.at(reshape_label).get_node_shared_ptr());

        const int left_shift = get_shapes_squeeze_shift(reshape->get_input_shape(0), reshape->get_output_shape(0));
        size_t gather_axis_value_current =
            convert_axis_to_positive(gather_axis->cast_vector<int64_t>()[0], gather->get_input_shape(0).size());
        size_t gather_axis_value_new = gather_axis_value_current - left_shift;

        auto gather_axis_new = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value_new);
        auto gather_indices_new = gather_indices->clone_with_new_inputs({});
        auto gather_new = std::make_shared<Gather>(reshape->input_value(0), gather_indices_new, gather_axis_new);

        auto reshape_const_new = reshape_const->clone_with_new_inputs({});
        auto reshape_new = reshape->clone_with_new_inputs({gather_new, reshape_const_new});

        replace_node_update_name(gather, reshape_new);
        copy_runtime_info(gather, {gather_new, gather_indices_new, gather_axis_new, reshape_new});

        register_new_node(gather_new);
        register_new_node(reshape_new);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
