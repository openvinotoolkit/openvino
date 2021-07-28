// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pdpd_fw_node.hpp>

namespace ngraph
{
    namespace frontend
    {
        NGRAPH_RTTI_DEFINITION(PDPDFrameworkNode, "PDPDFrameworkNode", 1);

        std::map<std::string, OutputVector> PDPDFrameworkNode::get_named_inputs() const
        {
            return m_decoder.map_for_each_input([&](const std::string&, size_t) {
                auto it = std::find(m_inputs_names.begin(), m_inputs_names.end(), name);
                if (it != m_inputs_names.end())
                {
                    return input(it - m_inputs_names.begin()).get_source_output();
                }
                else
                {
                    return Output<Node>();
                }
            });
        }

        std::map<std::string, OutputVector> PDPDFrameworkNode::get_named_outputs()
        {
            return m_decoder.map_for_each_output(
                [&](const std::string&, size_t idx) { return output(idx); });
        }

    } // namespace frontend
} // namespace ngraph
