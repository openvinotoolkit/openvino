// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/if.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/function.hpp"
#include "ngraph/runtime/reference/split.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            using namespace op::v8;
            enum if_body_indexes
            {
                then_body_index = 0,
                else_body_index = 1
            };
            void if_reference(
                const std::vector<std::shared_ptr<Function>>& bodies,
                const std::vector<op::util::MultiSubgraphOutputDescriptionVector>& out_descs,
                const std::vector<op::util::MultiSubgraphInputDescriptionVector>& input_descs,
                const HostTensorVector& out,
                const HostTensorVector& args)
            {
                auto condition_value = args[0]->get_data_ptr<bool>()[0];
                auto branch_index = (condition_value) ? if_body_indexes::then_body_index
                                                      : if_body_indexes::else_body_index;
                HostTensorVector inputs_to_body;
                HostTensorVector outs_from_body;
                // TODO: need find num of inputs in bode and reserve inputs_to_body
                inputs_to_body.resize(input_descs[branch_index].size());
                for (auto input_desc : input_descs[branch_index])
                {
                    inputs_to_body[input_desc->m_body_parameter_index] =
                        args[input_desc->m_input_index];
                }
                reference::function(bodies[branch_index], inputs_to_body, outs_from_body);
                for (auto out_descr : out_descs[branch_index])
                {
                    auto res = outs_from_body[out_descr->m_body_value_index];
                    out[out_descr->m_output_index]->write(res->get_data_ptr(),
                                                          res->get_size_in_bytes());
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph