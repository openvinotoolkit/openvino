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

#include "ie_backend_visibility.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ie
        {
            class IETensor : public ngraph::runtime::Tensor
            {
            public:
                IE_BACKEND_API IETensor(const ngraph::element::Type& element_type,
                                        const Shape& shape);
                IE_BACKEND_API IETensor(const ngraph::element::Type& element_type,
                                        const PartialShape& shape);

                ///
                /// \brief      Write bytes directly into the tensor
                ///
                /// \param      src    Pointer to source of data
                /// \param      bytes  Number of bytes to write, must be integral number of
                /// elements.
                ///
                void write(const void* src, size_t bytes) override;

                ///
                /// \brief      Read bytes directly from the tensor
                ///
                /// \param      dst    Pointer to destination for data
                /// \param      bytes  Number of bytes to read, must be integral number of elements.
                ///
                void read(void* dst, size_t bytes) const override;

                const void* get_data_ptr() const;

            private:
                IETensor(const IETensor&) = delete;
                IETensor(IETensor&&) = delete;
                IETensor& operator=(const IETensor&) = delete;
                AlignedBuffer m_data;
            };
        }
    }
}
