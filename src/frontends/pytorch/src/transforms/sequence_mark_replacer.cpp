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

        const auto inputs = seq_mark_node->get_sequence();

        if (inputs.empty()) {
            return false;
        }

        ov::pass::NodeRegistry rg;
        const auto neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});

        // Concatenate all inputs reshaped to 1D
        OutputVector inputs_to_concat;

        for (const auto& input : inputs) {
            // Flatten to 1D for consistent concatenation on axis 0.
            // Reshape(-1) handles scalars, 1D, and higher-rank tensors uniformly.
            // Don't call get_constant_from_source: it uses parameter upper
            // bounds, which default to 0 for unconstrained symint Parameter
            // inputs, and would turn [arg99_1, 2048] into [0, 2048].
            const auto reshape = rg.make<v1::Reshape>(input, neg_1, false);
            inputs_to_concat.push_back(reshape);
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
