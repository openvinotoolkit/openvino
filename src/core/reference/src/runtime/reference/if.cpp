// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/if.hpp"

#include "ngraph/op/if.hpp"
#include "ngraph/runtime/reference/function.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void if_reference(const std::vector<std::shared_ptr<Function>>& bodies,
                  const std::vector<op::util::MultiSubgraphOutputDescriptionVector>& out_descs,
                  const std::vector<op::util::MultiSubgraphInputDescriptionVector>& input_descs,
                  const HostTensorVector& out,
                  const HostTensorVector& args) {
    NGRAPH_CHECK(args.size() > 0, "If operation must have input condition value");

    auto condition_value = args[0]->get_data_ptr<bool>()[0];
    auto branch_index = (condition_value) ? op::v8::If::THEN_BODY_INDEX : op::v8::If::ELSE_BODY_INDEX;
    HostTensorVector inputs_to_body;
    HostTensorVector outs_from_body;
    inputs_to_body.resize(input_descs[branch_index].size());
    auto inputs_size = args.size();
    auto output_size = out.size();
    for (const auto& input_desc : input_descs[branch_index]) {
        NGRAPH_CHECK(inputs_size > input_desc->m_input_index,
                     "Incorrect associating! If has not input with id ",
                     input_desc->m_input_index);
        inputs_to_body[input_desc->m_body_parameter_index] = args[input_desc->m_input_index];
    }
    reference::function(bodies[branch_index], inputs_to_body, outs_from_body);
    for (const auto& out_descr : out_descs[branch_index]) {
        NGRAPH_CHECK(output_size > out_descr->m_output_index,
                     "Incorrect associating! If has not output with id ",
                     out_descr->m_output_index);
        auto res = outs_from_body[out_descr->m_body_value_index];
        out[out_descr->m_output_index]->set_shape(res->get_shape());
        out[out_descr->m_output_index]->write(res->get_data_ptr(), res->get_size_in_bytes());
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph