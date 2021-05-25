// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

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
                std::shared_ptr<ngraph::Node>
                    make_topk_subgraph(default_opset::TopK::Mode mode) const;

                const std::int64_t m_keep_dims;
                Output<ngraph::Node> m_input_node;
                std::int64_t m_axis;
                std::int64_t m_select_last_index;
            };

        } // namespace utils
    }     // namespace onnx_import
} // namespace ngraph
