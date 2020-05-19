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

#include <cstddef>
#include <vector>

#include "ngraph/descriptor/layout/tensor_layout.hpp"

namespace ngraph
{
    namespace descriptor
    {
        class Tensor;

        namespace layout
        {
            /// \brief The standard strided layout, used for row-major and column-major, their
            ///        permutations and slices.
            ///
            /// The linearized offset of an index I is dot(I, strides) + offset.
            class NGRAPH_API DenseTensorLayout : public TensorLayout
            {
            public:
                ~DenseTensorLayout() override {}
                DenseTensorLayout(const Tensor& tensor);

                size_t get_offset() const { return m_offset; }
                virtual size_t get_index_offset(const std::vector<size_t>& indices) override;
                Strides get_strides() const override;
                virtual bool operator==(const TensorLayout& other) const override;

            protected:
                size_t m_offset{0};
            };
        }
    }
}
