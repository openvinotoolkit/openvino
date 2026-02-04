// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sequence_mark_replacer.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

SequenceMarkReplacer::SequenceMarkReplacer() {
    const auto seq_mark_pattern = ov::pass::pattern::wrap_type<SequenceMark>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto seq_mark_node = ov::as_type_ptr<SequenceMark>(m.get_match_root());
        if (!seq_mark_node) {
            return false;
        }

        if (seq_mark_node->get_input_size() == 0) {
            return false;
        }

        ov::pass::NodeRegistry rg;
        const auto neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});

        // Concatenate all inputs reshaped to 1D
        OutputVector inputs_to_concat;

        for (const auto& input : seq_mark_node->input_values()) {
            const auto& input_rank = input.get_partial_shape().rank();

            if (input_rank.is_static() && input_rank.get_length() > 1) {
                // Elements with rank > 1 cannot be concatenated into 1D
                add_exception_to_fw_node(seq_mark_node, "unsupported SequenceMark: all inputs must be 0D or 1D.");
                return false;
            }

            // Reshape all elements to 1D for consistent concatenation
            const auto reshape = rg.make<v1::Reshape>(input, neg_1, false);
            if (const auto list_const = ov::util::get_constant_from_source(reshape)) {
                inputs_to_concat.push_back(list_const);
            } else {
                inputs_to_concat.push_back(reshape);
            }
        }

        const auto concat = rg.make<v0::Concat>(inputs_to_concat, 0);
        copy_runtime_info_and_name(seq_mark_node, rg.get());
        replace_node(seq_mark_node, concat);
        return true;
    };

    const auto m = std::make_shared<ov::pass::pattern::Matcher>(seq_mark_pattern,
                                                                "ov::frontend::pytorch::pass::SequenceMarkReplacer");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
