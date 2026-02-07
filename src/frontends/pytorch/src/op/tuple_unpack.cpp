// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_tuple_unpack(const NodeContext& context) {
    // ComplexTypeMark handling: unwrap input, create FrameworkNode with unwrapped data,
    // wrap outputs. This ensures transformations see clean graph without ComplexTypeMark.
    auto [input, complex] = unwrap_complex(context.get_input(0));

    if (const auto& seq_mark = ov::as_type_ptr<SequenceMark>(input.get_node_shared_ptr())) {
        // SequenceMark -> TupleUnpack can be annihilated
        auto res = seq_mark->get_sequence();
        return wrap_complex(context, res, complex);
    } else {
        // Create FrameworkNode with UNWRAPPED input (transformations will see clean graph)
        auto tuple_unpack_fw = std::make_shared<PtFrameworkNode>(context.get_decoder(),
                                                                 OutputVector{input},  // unwrapped!
                                                                 context.get_output_size());
        context.mark_node(tuple_unpack_fw);
        add_exception_to_fw_node(tuple_unpack_fw,
                                 "Tuples are not supported yet and can be resolved only in specific cases.");

        // Wrap outputs in ComplexTypeMark if input was complex
        return wrap_complex(context, tuple_unpack_fw->outputs(), complex);
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
