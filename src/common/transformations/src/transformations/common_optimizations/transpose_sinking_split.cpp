#include "transformations/common_optimizations/transpose_sinking_split.hpp"

#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov::pass::pattern;
using namespace ov;
using namespace ov::opset9;
using namespace transpose_sinking;

namespace {

using NodePtr = std::shared_ptr<Node>;

struct OutputTranspose {
    OutputTranspose() : transpose(nullptr), transpose_const(nullptr) {}
    Transpose* transpose;
    Constant* transpose_const;
};

OutputTranspose GetOutputTransposes(NodePtr node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            auto transpose_node = dynamic_cast<Transpose*>(input.get_node());
            if (!transpose_node)
                continue;
            auto constant_node = dynamic_cast<Constant*>(transpose_node->input_value(1).get_node());
            if (!constant_node)
                continue;
            {
                OutputTranspose output_transpose;
                output_transpose.transpose = transpose_node;
                output_transpose.transpose_const = constant_node;

                return output_transpose;
            }
        }
    }

    return {};
}

template <typename NodeT>
std::shared_ptr<ov::Node> FindInputNode(ov::Node* node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        std::shared_ptr<ov::Node> input_node = node->get_input_node_shared_ptr(input_idx);
        auto target_node = ov::as_type_ptr<NodeT>(input_node);
        if (target_node)
            return target_node;
    }
    return {};
}

bool HasInputSplitAndTransposeSiblings(const Output<Node>& output) {
    NodePtr main_node = FindInputNode<Split>(output.get_node());
    if (!main_node) {
        return false;
    }

    return HasSameOutputTransposeNodes(main_node);
}

bool IsSplitSinked(const Output<Node>& output) {
    return HasInputSplitAndTransposeSiblings(output) && is_sinking_node(output);
}

bool GetSplitAxis(const std::shared_ptr<Constant>& split_axis, const ov::Rank& rank, int64_t& axis) {
    auto split_axis_val = split_axis->cast_vector<int64_t>();
    if (split_axis_val.empty()) {
        return false;
    }
    axis = split_axis_val[0];
    if (axis < 0) {
        if (rank.is_static()) {
            const auto rank_val = rank.get_length();
            axis += rank_val;
        } else {
            return false;
        }
    }
    return true;
}
}  // namespace

/*
 * We follow Transpose operations rather than Split. We cannot create matcher pattern
 * for Split with Transpose outputs since Split can have different number of outputs.
 * We just can:
 * - specify Split as searched node and check if it has transpose outputs
 * - specify Transpose as searched node and check if it has Split input
 * Transformations are called on each found node in sorted order from the start to end
 * of the network. When we proceed Split backward sinking we move input transpose
 * to the input of the Split operation.
 * Consider case Split (1) -> Split (2) -> Transpose
 * If specify Split as main searched node after first transformation work we will have
 * Split (1) -> Transpose -> Split(2)
 * Matcher pass will not call TransposeSinkingSplitBackward since
 * - matcher pattern has no Transpose label
 * - Split (1) has already been proceeded
 * Adding Split(2) into the working queue as register_new_node(split)
 * cannot help us. We just can try to find all input Split operations and add them with
 * register_new_node(). Implemented way is simpler.
 *
 * We sink Transpose through Split operation in a backward way only if all the output
 * nodes are the same Transpose. We can:
 * - clone Split with all outputs except Transpose
 *   causes perfomance problems
 * - add reversed Transpose operations on all outputs except sinking Transpose
 *   nothing to do with new added output Transposes
 */
ov::pass::TransposeSinkingSplitBackward::TransposeSinkingSplitBackward() {
    MATCHER_SCOPE(TransposeSinkingSplitBackward);

    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_const_label}, IsSplitSinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_label_node = pattern_to_output.at(transpose_label).get_node();

        NodePtr split = FindInputNode<Split>(transpose_label_node);
        auto split_axis_constant = as_type_ptr<Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis_constant) {
            return false;
        }

        int64_t split_axis;
        if (!GetSplitAxis(split_axis_constant, split->input_value(0).get_partial_shape().rank(), split_axis)) {
            return false;
        }
        OutputTranspose output_transpose = GetOutputTransposes(split);

        const auto transpose_axis_order = output_transpose.transpose_const->get_axis_vector_val();
        const auto transpose_element_type = output_transpose.transpose_const->get_element_type();

        const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);

        const size_t reversed_transposed_split_axis = reversed_traspose_axis_order[split_axis];

        // insert transpose before split
        {
            auto input_node = split->input_value(0);
            auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                                  Shape{transpose_axis_order.size()},
                                                                  transpose_axis_order);
            auto new_transpose = std::make_shared<Transpose>(input_node, new_transpose_const);

            split->input(0).replace_source_output(new_transpose->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

            register_new_node(new_transpose);
        }

        // update split axis
        auto new_split_axis_const = std::make_shared<Constant>(split_axis_constant->get_element_type(),
                                                               Shape{},
                                                               reversed_transposed_split_axis);
        split->input(1).replace_source_output(new_split_axis_const);
        copy_runtime_info({split_axis_constant,
                           output_transpose.transpose->shared_from_this(),
                           output_transpose.transpose_const->shared_from_this()},
                          new_split_axis_const);

        // remove split output transposes
        RemoveSingleOutputConsumers(split);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeSinkingSplitForward::TransposeSinkingSplitForward() {
    MATCHER_SCOPE(TransposeSinkingSplitForward);

    auto main_node_label = wrap_type<Split>(IfNodeHasTransposeInputs);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();
        auto split = as_type_ptr<Split>(main_node);
        auto split_axis_constant = as_type_ptr<Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis_constant) {
            return false;
        }

        int64_t split_axis;
        if (!GetSplitAxis(split_axis_constant, split->input_value(0).get_partial_shape().rank(), split_axis)) {
            return false;
        }
        TransposeInputsInfo transpose_input_info = GetFirstTransposeInput(main_node);

        sink_forward::RemoveInputNode(main_node, /* input_idx */ 0);
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
            transpose_sinking::UpdateForwardSinkingAbility(new_node);
        }

        const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        const size_t transposed_split_axis = transpose_axis_order[split_axis];
        auto new_split_axis_const =
            std::make_shared<Constant>(split_axis_constant->get_element_type(), Shape{}, transposed_split_axis);
        split->input(1).replace_source_output(new_split_axis_const);
        copy_runtime_info({split_axis_constant, transpose_input_info.transpose, transpose_input_info.transpose_const},
                          new_split_axis_const);

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
