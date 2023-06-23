// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_split.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
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

using NodePtr = std::shared_ptr<Node>;

struct OutputGather {
    OutputGather() = default;
    Gather* gather = {};
    Constant* gather_indices = {};
    Constant* gather_axis = {};
    int output_idx = {};
};

OutputGather find_first_output_gather(NodePtr node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            auto gather_node = dynamic_cast<Gather*>(input.get_node());
            if (!gather_node)
                continue;
            auto indices_node = dynamic_cast<Constant*>(gather_node->input_value(1).get_node());
            if (!indices_node)
                continue;
            auto axis_node = dynamic_cast<Constant*>(gather_node->input_value(2).get_node());
            if (!axis_node)
                continue;
            {
                OutputGather output_gather;
                output_gather.gather = gather_node;
                output_gather.gather_indices = indices_node;
                output_gather.gather_axis = axis_node;
                output_gather.output_idx = static_cast<int>(output_idx);

                return output_gather;
            }
        }
    }

    return {};
}

}  // namespace

/*
 * We follow Gather operations rather than Split. We cannot create matcher pattern
 * for Split with Transpose outputs since Split can have different number of outputs.
 * We can just:
 * - specify Split as searched node and check if it has gather outputs
 * - specify Gather as searched node and check if it has Split input
 * Transformations are called on each found node in sorted order from the start to end
 * of the network. When we proceed Split backward sinking we move input Gather
 * to the input of the Split operation.
 * Consider case Split (1) -> Split (2) -> Gather
 * If specify Split as main searched node after first transformation work we will have
 * Split (1) -> Gather -> Split(2)
 * Matcher pass will not call GatherSinkingSplitBackward since
 * - matcher pattern has no Gather label
 * - Split (1) has already been proceeded
 * Adding Split(2) into the working queue as register_new_node(split)
 * cannot help us. We just can try to find all input Split operations and add them with
 * register_new_node(). Implemented way is simpler.
 */
GatherSinkingSplitBackward::GatherSinkingSplitBackward() {
    MATCHER_SCOPE(GatherSinkingSplitBackward);

    auto gather_indices_label = wrap_type<Constant>();
    auto gather_axis_label = wrap_type<Constant>();
    auto gather_label = wrap_type<Gather>({any_input(), gather_indices_label, gather_axis_label}, is_split_sinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());

        auto split = find_first_input_node<Split>(gather);
        auto split_axis_constant = as_type_ptr<Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis_constant) {
            return false;
        }

        int64_t split_axis;
        if (!get_split_axis(split_axis_constant, split->input_value(0).get_partial_shape().rank(), split_axis)) {
            return false;
        }

        OutputGather output_gather = find_first_output_gather(split);
        const int64_t gather_axis =
            convert_axis_to_positive(output_gather.gather_axis->cast_vector<int64_t>()[0],
                                     output_gather.gather->input(0).get_partial_shape().rank().get_length());

        const auto gather_indices = normalize_gather_indices(output_gather.gather_indices->cast_vector<int64_t>());
        std::vector<int64_t> new_indices(split->get_input_shape(0)[gather_axis]);
        std::iota(new_indices.begin(), new_indices.end(), 0);

        const size_t base = output_gather.output_idx * split->get_output_shape(0)[gather_axis];
        for (size_t i = 0; i < gather_indices.size(); ++i) {
            new_indices[base + i] = base + gather_indices[i];
        }

        auto split_input = split->input_value(0);
        auto new_indices_const = std::make_shared<Constant>(output_gather.gather_axis->get_element_type(),
                                                            Shape{new_indices.size()},
                                                            new_indices);
        auto new_axis_const = output_gather.gather_axis->clone_with_new_inputs({});
        auto new_gather = std::make_shared<Gather>(split_input, new_indices_const, new_axis_const);
        split->input(0).replace_source_output(new_gather->output(0));
        copy_runtime_info(split_input.get_node_shared_ptr(), {new_gather, new_indices_const, new_axis_const});
        register_new_node(new_gather);

        for (auto& input : split->get_output_target_inputs(output_gather.output_idx)) {
            Node* consumer = input.get_node();
            if (consumer->get_output_size() != 1)
                continue;
            consumer->output(0).replace(split->output(output_gather.output_idx));
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
