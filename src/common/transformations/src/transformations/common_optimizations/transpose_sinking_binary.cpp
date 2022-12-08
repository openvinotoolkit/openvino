#include "transformations/common_optimizations/transpose_sinking_binary.hpp"

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

ov::pass::TransposeSinkingBinaryElementwiseForward::TransposeSinkingBinaryElementwiseForward() {
    MATCHER_SCOPE(TransposeSinkingBinaryElementwiseForward);

    auto main_node_label = wrap_type<op::util::BinaryElementwiseArithmetic>(IfNodeHasTransposeInputs);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();

        TransposeInputsInfo transpose_input_info = GetFirstTransposeInput(main_node);

        sink_forward::UpdateInputTransposes(main_node, transpose_input_info);
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
            transpose_sinking::UpdateForwardSinkingAbility(new_node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

pass::TransposeSinkingBinaryElementwiseBackward::TransposeSinkingBinaryElementwiseBackward() {
    MATCHER_SCOPE(TransposeSinkingBinaryElementwiseBackward);

    auto main_node_label = wrap_type<op::util::BinaryElementwiseArithmetic>(consumers_count(1));

    auto transpose_const_label = wrap_type<Constant>(consumers_count(1));

    auto IfSinkingEnabled = [](const Output<Node>& output) -> bool {
        static auto consumers_check = consumers_count(1);
        return consumers_check(output) && is_sinking_node(output.get_node_shared_ptr());
    };

    auto transpose_label = wrap_type<Transpose>({main_node_label, transpose_const_label}, IfSinkingEnabled);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, transpose_const)) {
            register_new_node(new_node);
        }

        // remove transpose after main node
        transpose->output(0).replace(main_node);

        SwapNames(transpose, main_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
