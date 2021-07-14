// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn.hpp"
#include <ngraph/opsets/opset6.hpp>
#include "lstm.hpp"
#include "paddlepaddle_frontend/utility.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs rnn(const NodeContext& node)
                {
                    auto mode = node.get_attribute<std::string>("mode");
                    PDPD_ASSERT(mode == "LSTM", "RNN only support LSTM now");
                    return lstm(node);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
