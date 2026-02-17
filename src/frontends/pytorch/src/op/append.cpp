// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_append(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto seq = context.get_input(0);
    auto tensor = context.get_input(1);
    auto seq_insert = std::make_shared<ov::frontend::SequenceInsert>(seq, tensor);
    context.mark_node(seq_insert);
    auto result = context.mark_node(std::make_shared<ov::frontend::SequenceMark>(OutputVector{seq_insert}));
    // aten::append mutates input list in-place. Update the tensor map so that
    // subsequent reads of the same tensor ID (e.g. by aten::cat) see the
    // chained SequenceInsert rather than the original SequenceMark.
    // Inside a Loop body this also causes an extra Result to be created,
    // turning the list into a carried (merged) value that SequenceConcatReplacer
    // can rewrite.
    context.mutate_input(0, result);
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
