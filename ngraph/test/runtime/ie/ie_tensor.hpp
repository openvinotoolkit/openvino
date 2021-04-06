// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
