// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/if.hpp"

#include "openvino/op/if.hpp"
#include "openvino/reference/function.hpp"

namespace ov {
namespace reference {
void if_reference(const std::vector<std::shared_ptr<Model>>& bodies,
                  const std::vector<op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector>& out_descs,
                  const std::vector<op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector>& input_descs,
                  ov::TensorVector& out,
                  const ov::TensorVector& args) {
    OPENVINO_ASSERT(args.size() > 0, "If operation must have input condition value");

    auto condition_value = args[0].data<bool>()[0];
    auto branch_index = (condition_value) ? op::v8::If::THEN_BODY_INDEX : op::v8::If::ELSE_BODY_INDEX;
    ov::TensorVector inputs_to_body;
    ov::TensorVector outs_from_body;
    inputs_to_body.resize(input_descs[branch_index].size());
    auto inputs_size = args.size();
    auto output_size = out.size();
    for (const auto& input_desc : input_descs[branch_index]) {
        OPENVINO_ASSERT(inputs_size > input_desc->m_input_index,
                        "Incorrect associating! If has not input with id ",
                        input_desc->m_input_index);
        inputs_to_body[input_desc->m_body_parameter_index] = args[input_desc->m_input_index];
    }
    reference::function(bodies[branch_index], inputs_to_body, outs_from_body);
    for (const auto& out_descr : out_descs[branch_index]) {
        OPENVINO_ASSERT(output_size > out_descr->m_output_index,
                        "Incorrect associating! If has not output with id ",
                        out_descr->m_output_index);
        const auto& res = outs_from_body[out_descr->m_body_value_index];
        res.copy_to(out[out_descr->m_output_index]);
    }
}
}  // namespace reference
}  // namespace ov
