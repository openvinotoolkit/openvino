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

#include <cstdint>
#include <memory>

#include "core/node.hpp"
#include "default_opset.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
        {
            /// \brief  Factory class which generates sub-graphs for ONNX ArgMin, ArgMax ops.
            class ArgMinMaxFactory
            {
            public:
                explicit ArgMinMaxFactory(const Node& node);
                virtual ~ArgMinMaxFactory() = default;

                /// \brief      Creates ArgMax ONNX operation.
                /// \return     Sub-graph representing ArgMax op.
                std::shared_ptr<ngraph::Node> make_arg_max() const;

                /// \brief      Creates ArgMin ONNX operation.
                /// \return     Sub-graph representing ArgMin op.
                std::shared_ptr<ngraph::Node> make_arg_min() const;

            private:
                Output<ngraph::Node>
                    make_topk_subgraph(default_opset::TopK::Mode mode) const;

                const std::int64_t m_keep_dims;
                std::shared_ptr<ngraph::Node> m_input_node;
                std::int64_t m_axis;
            };

        } // namespace arg
    }     // namespace onnx_import
} // namespace ngraph
