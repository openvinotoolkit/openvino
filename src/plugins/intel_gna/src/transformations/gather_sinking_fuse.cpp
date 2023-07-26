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

struct TransformationInfo {
    std::shared_ptr<Constant> input_indices_const;
    std::shared_ptr<Constant> input_axis_const;
    std::shared_ptr<Gather> input_gather;
    std::shared_ptr<Constant> output_indices_const;
    std::shared_ptr<Constant> output_axis_const;
    std::shared_ptr<Gather> output_gather;
};

std::shared_ptr<Gather> fuse_gather_nodes(TransformationInfo& info) {
    const std::vector<int64_t> input_gather_indices = get_normalized_gather_indices(info.input_indices_const);
    const std::vector<int64_t> output_gather_indices = get_normalized_gather_indices(info.output_indices_const);
    const std::vector<int64_t> result_gather_indices =
        combine_gather_indexes(input_gather_indices, output_gather_indices);
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

inline bool is_skip_operation(const std::shared_ptr<ov::Node>& node) {
    return (std::dynamic_pointer_cast<Reshape>(node) != nullptr ||
            std::dynamic_pointer_cast<Squeeze>(node) != nullptr ||
            std::dynamic_pointer_cast<Unsqueeze>(node) != nullptr) &&
           has_n_consumers(node, 1);
}

}  // namespace

GatherSinkingFuse::GatherSinkingFuse() {
    MATCHER_SCOPE(GatherSinkingFuse);

    auto indices_in_const_label = wrap_type<Constant>(is_constant_1d);
    auto axis_in_const_label = wrap_type<Constant>(is_constant_1d);
    auto gather_in_label = wrap_type<Gather>({any_input(), indices_in_const_label, axis_in_const_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather_in = as_type_ptr<Gather>(pattern_to_output.at(gather_in_label).get_node_shared_ptr());

        TransformationInfo info;
        info.output_indices_const =
            as_type_ptr<Constant>(pattern_to_output.at(indices_in_const_label).get_node_shared_ptr());
        info.output_axis_const = as_type_ptr<Constant>(pattern_to_output.at(axis_in_const_label).get_node_shared_ptr());
        info.output_gather = gather_in;

        // skip all the non-functional layers
        std::shared_ptr<ov::Node> non_reshape_node =
            graph_utils::get_prev_node_skipping_certain(gather_in->get_input_node_shared_ptr(0), is_skip_operation);
        auto gather_out = std::dynamic_pointer_cast<Gather>(non_reshape_node);
        if (!gather_out) {
            return false;
        }

        info.input_indices_const = as_type_ptr<Constant>(gather_out->get_input_node_shared_ptr(1));
        info.input_axis_const = as_type_ptr<Constant>(gather_out->get_input_node_shared_ptr(2));
        info.input_gather = gather_out;

        auto new_node = fuse_gather_nodes(info);
        if (new_node)
            register_new_node(new_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_in_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
