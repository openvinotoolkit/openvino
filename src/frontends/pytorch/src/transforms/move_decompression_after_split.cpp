// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_decompression_after_split.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

namespace ov::frontend::pytorch::pass {

using namespace ov::op;
using namespace ov::pass::pattern;

MoveDecompressionAfterSplit::MoveDecompressionAfterSplit() {
    // Folding addresses bytes, so packed sub-byte storage is skipped.
    const auto constant = wrap_type<v0::Constant>([](const Output<Node>& output) {
        const auto bitwidth = output.get_element_type().bitwidth();
        return bitwidth >= 8 && bitwidth % 8 == 0;
    });
    const auto convert = wrap_type<v0::Convert>({constant}, [](const Output<Node>& output) {
        return ov::is_decompression(output.get_node_shared_ptr());
    });
    const auto axis = any_input();
    const auto split = wrap_type<v1::Split>({convert, axis});
    const auto variadic_split = wrap_type<v1::VariadicSplit>({convert, axis, any_input()});
    const auto split_pattern = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{split, variadic_split});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto split_node = m.get_match_root();
        if (transformation_callback(split_node) || ov::pass::constant_folding_is_disabled(split_node))
            return false;
        const auto convert_node = m.get_pattern_value_map().at(convert).get_node_shared_ptr();

        OutputVector inputs{convert_node->input_value(0)};
        for (size_t i = 1; i < split_node->get_input_size(); ++i) {
            auto input = ov::util::get_constant_from_source(split_node->input_value(i));
            if (!input)
                return false;
            inputs.push_back(input);
        }
        const auto low_precision_split = split_node->clone_with_new_inputs(inputs);
        OutputVector folded(split_node->get_output_size());
        if (!low_precision_split->constant_fold(folded, inputs))
            return false;

        OutputVector replacements;
        for (size_t i = 0; i < folded.size(); ++i) {
            const auto folded_node = folded[i].get_node_shared_ptr();
            auto replacement = convert_node->clone_with_new_inputs({folded[i]});
            replacement->set_friendly_name(split_node->get_friendly_name() + "." + std::to_string(i));
            ov::copy_runtime_info({inputs[0].get_node_shared_ptr(), split_node}, folded_node);
            ov::copy_runtime_info({convert_node, split_node}, replacement);
            // Decompression and constant-folding markers are not copyable rt_info.
            ov::mark_as_decompression(replacement);
            ov::pass::disable_constant_folding(replacement);
            ov::pass::disable_constant_folding(folded_node);
            replacements.push_back(replacement);
        }
        ov::replace_node(split_node, replacements);
        return true;
    };

    register_matcher(
        std::make_shared<Matcher>(split_pattern, "ov::frontend::pytorch::pass::MoveDecompressionAfterSplit"),
        callback);
}

}  // namespace ov::frontend::pytorch::pass
