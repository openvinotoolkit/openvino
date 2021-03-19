//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            class NGRAPH_API ArithmeticReductionKeepDims : public util::ArithmeticReduction
            {
            protected:
                ArithmeticReductionKeepDims() = default;

                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ArithmeticReductionKeepDims(const Output<Node>& arg,
                                            const Output<Node>& reduction_axes,
                                            bool keep_dims = false);

                bool visit_attributes(AttributeVisitor& visitor) override;

            public:
                void validate_and_infer_types() override;

                /// \return If set to 1 it holds axes that are used for reduction.
                /// For each such axis, output dimension is equal to 1.
                bool get_keep_dims() const { return m_keep_dims; }
                void set_keep_dims(bool keep_dims) { m_keep_dims = keep_dims; }

            private:
                bool m_keep_dims = false;
            };
        }
    }
}
