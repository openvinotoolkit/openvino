// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime tensor API
 *
 * @file openvino/runtime/tensor.hpp
 */
#pragma once

#include <type_traits>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"

namespace InferenceEngine {
class Blob;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {

class InferRequest;
class RemoteContext;
class VariableState;

/**
 * @brief Tensor API holding host memory
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class OPENVINO_API Tensor {
protected:
    std::shared_ptr<void> _so;                     //!< Reference to dynamicly loaded library
    std::shared_ptr<InferenceEngine::Blob> _impl;  //!< Shared pointer to internal tensor representation

    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that Tensor can work properly even if plugin object is
     * destroyed.
     * @param impl Initialized shared pointer
     */
    Tensor(const std::shared_ptr<void>& so, const std::shared_ptr<InferenceEngine::Blob>& impl);

    friend class ov::runtime::InferRequest;
    friend class ov::runtime::RemoteContext;
    friend class ov::runtime::VariableState;

public:
    using Ptr = std::shared_ptr<Tensor>;
    /**
     * @brief Default constructor
     */
    Tensor() = default;
    virtual ~Tensor() = default;

    /**
     * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param allocator allocates memory for internal tensor storage
     */
    Tensor(const element::Type type, const Shape& shape, const Allocator& allocator = {});

    /**
     * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
     * @note Does not perform memory allocation internally
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param host_ptr Pointer to pre-allocated host memory
     * @param strides Optional strides parameters in elements. Strides are supposed to be equal to shape if they are not
     * set
     */
    Tensor(const element::Type type, const Shape& shape, void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs region of interest (ROI) tensor form another tensor.
     * @note Does not perform memory allocation internally
     * @param other original tensor
     * @param begin start coordinate of ROI object inside of the original object.
     * @param end end coordinate of ROI object inside of the original object.
     * @note A Number of dimensions in `begin` and `end` must match number of dimensions in `other.get_shape()`
     */
    Tensor(const Tensor& other, const Coordinate& begin, const Coordinate& end);

    /**
     * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
     * @note Memory allocation may happen
     * @param shape A new shape
     */
    void set_shape(const ov::Shape& shape);

    /**
     * @return A tensor element type
     */
    element::Type get_element_type() const;

    /**
     * @return A tensor shape
     */
    Shape get_shape() const;

    /**
     * @brief Returns the total number of elements (a product of all the dims or 1 for scalar)
     * @return The total number of elements
     */
    size_t get_size() const;

    /**
     * @brief Returns the size of the current Tensor in bytes.
     * @return Tensor's size in bytes
     */
    size_t get_byte_size() const;

    /**
     * @return Tensor's strides in elements
     */
    Strides get_strides() const;

    /**
     * @brief Provides an access to the underlaying host memory
     * @param type Optional type parameter.
     * @note If type parameter is specified, the method throws an exception
     * if specified type's fundamental type does not match with tensor element type's fundamental type
     * @return A host pointer to tensor memory
     */
    void* data(const element::Type type = {}) const;

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
     * @brief Checks if current Tensor object is not initialized
     * @return `true` if current Tensor object is not initialized, `false` - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Tensor object is initialized
     * @return `true` if current Tensor object is initialized, `false` - otherwise
     */
    explicit operator bool() const noexcept;
};

using TensorVector = std::vector<Tensor::Ptr>;
}  // namespace runtime
}  // namespace ov
