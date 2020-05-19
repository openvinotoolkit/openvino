//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <utility>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API QuantizedDotBias : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"QuantizedDotBias", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                QuantizedDotBias() = default;
                QuantizedDotBias(const Output<Node>& data,
                                 const Output<Node>& weights,
                                 const Output<Node>& bias,
                                 const Output<Node>& scale,
                                 bool requantize = true,
                                 bool with_relu = false);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override
                {
                    check_new_args_count(this, new_args);
                    return std::make_shared<QuantizedDotBias>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3),
                                                              m_requantize,
                                                              m_with_relu);
                }
                bool with_relu() const { return m_with_relu; }
                bool requantize() const { return m_requantize; }
            protected:
                bool m_requantize;
                bool m_with_relu;
            };
        }
        using v0::QuantizedDotBias;
    }
}
