// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"

namespace ov {

class OPENVINO_API ITensor : public std::enable_shared_from_this<ITensor> {
public:
    /**
     * @brief Set new shape for tensor
     * @note Memory allocation may happen
     * @param shape A new shape
     */
    virtual void set_shape(ov::Shape shape) = 0;

    /**
     * @return A tensor element type
     */
    virtual const ov::element::Type& get_element_type() const = 0;

    /**
     * @return A tensor shape
     */
    virtual const ov::Shape& get_shape() const = 0;

    /**
     * @brief Returns the total number of elements (a product of all the dims or 1 for scalar)
     * @return The total number of elements
     */
    virtual size_t get_size() const;

    /**
     * @brief Returns the size of the current Tensor in bytes.
     * @return Tensor's size in bytes
     */
    virtual size_t get_byte_size() const;

    /**
     * @return Tensor's strides in bytes
     */
    virtual const ov::Strides& get_strides() const = 0;

    /**
     * @brief Provides an access to the underlaying host memory
     * @param type Optional type parameter.
     * @note If type parameter is specified, the method throws an exception
     * if specified type's fundamental type does not match with tensor element type's fundamental type
     * @return A host pointer to tensor memory
     */
    virtual void* data(const element::Type& type = {}) const = 0;

    /**
     * @brief Provides an access to the underlaying host memory casted to type `T`
     * @return A host pointer to tensor memory casted to specified type `T`.
     * @note Throws exception if specified type does not match with tensor element type
     */
    template <typename T, typename datatype = typename std::decay<T>::type>
    T* data() const {
        return static_cast<T*>(data(element::from<datatype>()));
    }

    /**
     * @brief Reports whether the tensor is continuous or not
     *
     * @return true if tensor is continuous
     */
    bool is_continuous() const;

    /**
     * @brief Copy tensor, destination tensor should have the same element type and shape
     *
     * @param dst destination tensor
     */
    virtual void copy_to(const std::shared_ptr<ov::ITensor>& dst) const;

protected:
    virtual ~ITensor();
};

}  // namespace ov
