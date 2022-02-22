// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace runtime {
class NGRAPH_API Tensor {
protected:
    Tensor(const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor) : m_descriptor(descriptor), m_stale(true) {}

public:
    virtual ~Tensor() {}
    Tensor& operator=(const Tensor&) = default;

    /// \brief Get tensor shape
    /// \return const reference to a Shape
    virtual const ngraph::Shape& get_shape() const;

    /// \brief Get tensor partial shape
    /// \return const reference to a PartialShape
    const ngraph::PartialShape& get_partial_shape() const;

    /// \brief Get tensor element type
    /// \return element::Type
    virtual const element::Type& get_element_type() const;

    /// \brief Get number of elements in the tensor
    /// \return number of elements in the tensor
    virtual size_t get_element_count() const;

    /// \brief Get the size in bytes of the tensor
    /// \return number of bytes in tensor's allocation
    virtual size_t get_size_in_bytes() const;

    /// \brief Get tensor's unique name
    /// \return tensor's name
    NGRAPH_DEPRECATED("Only output ports have names")
    const std::string& get_name() const;

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param n Number of bytes to write, must be integral number of elements.
    virtual void write(const void* p, size_t n) = 0;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param n Number of bytes to read, must be integral number of elements.
    virtual void read(void* p, size_t n) const = 0;

protected:
    std::shared_ptr<ngraph::descriptor::Tensor> m_descriptor;
    bool m_stale;
};
}  // namespace runtime
}  // namespace ngraph
