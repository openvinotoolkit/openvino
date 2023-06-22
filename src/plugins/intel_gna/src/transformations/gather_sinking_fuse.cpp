// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_fuse.hpp"

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
bool is_gather_with_parent_gather_same_axis(const Output<Node>& output) {
    int64_t output_gather_axis = {};
    if (!get_gather_axis(output.get_node_shared_ptr(), output_gather_axis))
        return false;
    int64_t input_gather_axis = {};
    if (!get_gather_axis(output.get_node_shared_ptr()->input_value(0).get_node_shared_ptr(), input_gather_axis))
        return false;
    return input_gather_axis == output_gather_axis;
}

struct TransformationInfo {
    std::shared_ptr<Constant> input_indices_const;
    std::shared_ptr<Constant> input_axis_const;
    std::shared_ptr<Gather> input_gather;
    std::shared_ptr<Constant> output_indices_const;
    std::shared_ptr<Constant> output_axis_const;
    std::shared_ptr<Gather> output_gather;
};

std::vector<int64_t> combine_gather_permutations(const std::vector<int64_t>& input_gather_indices,
                                                 const std::vector<int64_t>& output_gather_indices) {
    if (input_gather_indices.size() != output_gather_indices.size())
        return {};
    std::vector<int64_t> result(input_gather_indices.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = input_gather_indices[output_gather_indices[i]];
    }

    return result;
}

std::shared_ptr<Gather> fuse_gather_nodes(TransformationInfo& info) {
    const std::vector<int64_t> input_gather_indices = get_normalized_gather_indices(info.input_indices_const);
    const std::vector<int64_t> output_gather_indices = get_normalized_gather_indices(info.output_indices_const);
    const std::vector<int64_t> result_gather_indices =
        combine_gather_permutations(input_gather_indices, output_gather_indices);
    if (is_pointless_permutation(result_gather_indices)) {
        ov::replace_output_update_name(info.output_gather->output(0), info.input_gather->input_value(0));
        return {};
    }

    const auto indices_element_type = info.output_axis_const->get_element_type();
    auto new_indices_const =
        std::make_shared<Constant>(indices_element_type, Shape{result_gather_indices.size()}, result_gather_indices);
    auto new_axis_const = info.output_axis_const->clone_with_new_inputs({});
    auto new_gather = std::make_shared<Gather>(info.input_gather->input_value(0), new_indices_const, new_axis_const);

    ov::replace_node(info.output_gather, new_gather);
    copy_runtime_info(info.input_gather, {new_gather, new_indices_const, new_axis_const});
    new_gather->set_friendly_name(info.output_gather->get_friendly_name());

    return new_gather;
}
}  // namespace

GatherSinkingFuse::GatherSinkingFuse() {
    MATCHER_SCOPE(GatherSinkingFuse);

    auto input_indices_const_label = wrap_type<Constant>(is_constant_1d);
    auto input_axis_const_label = wrap_type<Constant>(is_constant_1d);
    auto input_gather_label = wrap_type<Gather>({any_input(), input_indices_const_label, input_axis_const_label});
    auto output_indices_const_label = wrap_type<Constant>(is_constant_1d);
    auto output_axis_const_label = wrap_type<Constant>(is_constant_1d);
    auto output_gather_label =
        wrap_type<Gather>({input_gather_label, output_indices_const_label, output_axis_const_label},
                          is_gather_with_parent_gather_same_axis);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        TransformationInfo info;
        info.input_indices_const =
            as_type_ptr<Constant>(pattern_to_output.at(input_indices_const_label).get_node_shared_ptr());
        info.input_axis_const =
            as_type_ptr<Constant>(pattern_to_output.at(input_axis_const_label).get_node_shared_ptr());
        info.input_gather = as_type_ptr<Gather>(pattern_to_output.at(input_gather_label).get_node_shared_ptr());
        info.output_indices_const =
            as_type_ptr<Constant>(pattern_to_output.at(output_indices_const_label).get_node_shared_ptr());
        info.output_axis_const =
            as_type_ptr<Constant>(pattern_to_output.at(output_axis_const_label).get_node_shared_ptr());
        info.output_gather = as_type_ptr<Gather>(pattern_to_output.at(output_gather_label).get_node_shared_ptr());

        auto new_node = fuse_gather_nodes(info);
        if (new_node)
            register_new_node(new_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(output_gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
