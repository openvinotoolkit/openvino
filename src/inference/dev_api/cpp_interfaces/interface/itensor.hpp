// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for Blob and generic TBlob<>
 *
 * @file ie_blob.h
 */
#pragma once

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/allocator.hpp"

namespace InferenceEngine {
class Blob;
}  // namespace InferenceEngine

namespace ov {

/**
 * @brief This class represents a universal container in the Inference Engine
 *
 * @note Each ITensor implementation must be derived from this ITensor class directly or indirectly
 */
class OPENVINO_API ITensor : public std::enable_shared_from_this<ITensor> {
public:
    /**
     * @brief A smart pointer containing ITensor object
     */
    using Ptr = std::shared_ptr<ITensor>;

    /**
     * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
     * @note Memory allocation may happen
     * @param shape A new shape
     */
    virtual void set_shape(const ov::Shape& shape);

    /**
     * @return A tensor element type
     */
    virtual element::Type get_element_type() const;

    /**
     * @return A tensor shape
     */
    virtual Shape get_shape() const;

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
     * @return Tensor's offseta in elements
     */
    virtual Coordinate get_offsets() const;

    /**
     * @return Tensor's strides in bytes
     */
    virtual Strides get_strides() const;

    /**
     * @brief Provides an access to the underlaying host memory
     * @param type Optional type parameter.
     * @note If type parameter is specified, the method throws an exception
     * if specified type's fundamental type does not match with tensor element type's fundamental type
     * @return A host pointer to tensor memory
     */
    virtual void* data(const element::Type type = {}) const;

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
     * @brief Return additional information associated with tensor
     * @return Map of property names to properties
     */
    virtual AnyMap get_properties() const;

protected:
    ~ITensor() = default;
};

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param allocator allocates memory for internal tensor storage
 */
OPENVINO_API ITensor::Ptr make_tensor(const element::Type type, const Shape& shape, const Allocator& allocator = {});

/**
 * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
 * @note Does not perform memory allocation internally
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
 * on shape and element size
 */
OPENVINO_API ITensor::Ptr make_tensor(const element::Type type, const Shape& shape, void* host_ptr, const Strides& strides = {});

/**
 * @brief Constructs region of interest (ROI) tensor form another tensor.
 * @note Does not perform memory allocation internally
 * @param other original tensor
 * @param begin start coordinate of ROI object inside of the original object.
 * @param end end coordinate of ROI object inside of the original object.
 * @note A Number of dimensions in `begin` and `end` must match number of dimensions in `other.get_shape()`
 */
OPENVINO_API ITensor::Ptr make_tensor(const ITensor::Ptr& other, const Coordinate& begin, const Coordinate& end);

/** @cond INTERNAL */
OPENVINO_API std::shared_ptr<ie::Blob> tensor_to_blob(const ITensor::Ptr& tensor);

OPENVINO_API ITensor::Ptr blob_to_tensor(const std::shared_ptr<ie::Blob>& tensor);
/** @endcond */

}  // namespace ov
