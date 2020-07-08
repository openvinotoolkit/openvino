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

#include <memory>
#include <type_traits>

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace pooling
        {
            ///
            /// \brief      Factory class which generates sub-graphs for ONNX 'regular' pooling
            ///             operators.
            ///
            /// \note       This factory is intended for creating pooling operations like:
            ///             - AveragePool
            ///             - MaxPool
            ///
            ///             This base class holds all common attributes like srides, dilations,
            ///             paddings, kernel shape and auto_pad type.
            ///
            /// \see        GlobalPoolingFactory
            class PoolingFactory
            {
            public:
                virtual ~PoolingFactory() = default;

                ///
                /// \brief      Creates average pooling ONNX operation.
                /// \return     Vector of output nodes.
                ///
                OutputVector make_avg_pool() const;

                ///
                /// \brief      Creates max pooling ONNX operation.
                /// \return     Vector of output nodes.
                ///
                OutputVector make_max_pool() const;

            protected:
                explicit PoolingFactory(const Node& node);

                Node m_onnx_node;
                const OutputVector m_inputs;
                Shape m_kernel_shape;
                Strides m_strides;
                Strides m_dilations;
                Shape m_padding_below;
                Shape m_padding_above;
                ngraph::op::PadType m_auto_pad;
            };

            ///
            /// \brief      Factory class which generates sub-graphs for ONNX 'local' pooling
            ///             operators.
            /// \note       For a 'local' pooling operation, the kernel shape attribute is required
            class LocalPoolingFactory : public PoolingFactory
            {
            public:
                explicit LocalPoolingFactory(const Node& node);
                virtual ~LocalPoolingFactory() = default;
            };

            ///
            /// \brief      Factory class which generates sub-graphs for ONNX 'global' pooling
            ///             operators.
            /// \note       In a 'global' pooling operation, the kernel shape is calculated
            ///             based on spatial dims
            class GlobalPoolingFactory : public PoolingFactory
            {
            public:
                explicit GlobalPoolingFactory(const Node& node);
                virtual ~GlobalPoolingFactory() = default;
            };

        } // namespace pooling
    }     // namespace onnx_import
} // namespace ngraph
