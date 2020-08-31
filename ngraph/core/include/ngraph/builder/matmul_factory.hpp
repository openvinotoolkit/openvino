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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        NGRAPH_SUPPRESS_DEPRECATED_START

        /// \brief  Factory class which generates an nGraph sub-graph performing MatMul operation.
        ///
        /// This default implementation `MatmulFactory` creates a `MatMul` operation for
        /// floating-point data.
        /// Subclasses: `QLinearMatmulFactory` and `MatmulIntegerFactory` implement quantized
        /// versions.
        class NGRAPH_DEPRECATED("This builder was deprecated.") NGRAPH_API MatmulFactory
        {
        public:
            explicit MatmulFactory(const OutputVector& inputs)
                : m_inputs(inputs)
            {
            }

            virtual ~MatmulFactory() = default;

            /// \brief Create a sub-graph representing an ONNX MatMul operation.
            ///
            /// \return OutputVector containing the sub-graph output node.
            virtual OutputVector make_matmul_op();

        protected:
            /// \return Output representing the left operand.
            virtual Output<Node> get_left();

            /// \return Output representing the right operand.
            virtual Output<Node> get_right();

            /// \return Output representing the nGraph Dot operation used to construct MatMul.
            virtual Output<Node> make_dot(const Output<Node>& left, const Output<Node>& right);

            const OutputVector m_inputs;
        };

        /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX QLinearMatMul
        /// operation.
        class NGRAPH_DEPRECATED("This builder was deprecated.") NGRAPH_API QLinearMatmulFactory
            : public MatmulFactory
        {
        public:
            explicit QLinearMatmulFactory(const OutputVector& inputs)
                : MatmulFactory(inputs)
            {
            }

        protected:
            Output<Node> get_right() override;
            Output<Node> make_dot(const Output<Node>& left, const Output<Node>& right) override;
        };

        /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX MatMulInteger
        /// operation.
        class NGRAPH_DEPRECATED("This builder was deprecated.") NGRAPH_API MatmulIntegerFactory
            : public MatmulFactory
        {
        public:
            explicit MatmulIntegerFactory(const OutputVector& inputs)
                : MatmulFactory(inputs)
            {
            }

        protected:
            Output<Node> make_dot(const Output<Node>& left, const Output<Node>& right) override;
        };

        NGRAPH_SUPPRESS_DEPRECATED_END
    } // namespace builder
} // namespace ngraph
