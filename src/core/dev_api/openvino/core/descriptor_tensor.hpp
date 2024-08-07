// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <unordered_set>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace descriptor {

class Tensor;

// To change Tensor element type please change the Parameter type.
void set_element_type(Tensor& tensor, const element::Type& elemenet_type);

// To change Tensor type please change the Parameter type.
void set_tensor_type(Tensor& tensor, const element::Type& element_type, const PartialShape& pshape);

OPENVINO_DEPRECATED("get_ov_tensor_legacy_name() is deprecated. Please don't use this function.")
OPENVINO_API
std::string get_ov_tensor_legacy_name(const Tensor& tensor);

OPENVINO_DEPRECATED("set_ov_tensor_legacy_name() is deprecated. Please don't use this function.")
OPENVINO_API
void set_ov_tensor_legacy_name(Tensor& tensor, const std::string& tensor_name);

/** @brief Tensor descriptor interface. */
class OPENVINO_API ITensorDescriptor {
public:
    virtual const element::Type& get_element_type() const = 0;
    virtual const PartialShape& get_partial_shape() const = 0;
    virtual const Shape& get_shape() const = 0;
    virtual void set_type_shape(const element::Type& et, const PartialShape& shape) = 0;

    virtual void set_names(const std::unordered_set<std::string>& names) = 0;
    virtual void add_names(const std::unordered_set<std::string>& names) = 0;
    virtual const std::unordered_set<std::string>& get_names() const = 0;
    virtual const std::string& get_any_name() const = 0;

    virtual RTMap& rt_map() = 0;
    virtual const RTMap& rt_map() const = 0;

    // Legacy name compatibility API
    OPENVINO_DEPRECATED("The legacy_name() is deprecated. Please don't use it")
    virtual std::string& legacy_name() = 0;
    OPENVINO_DEPRECATED("The legacy_name() is deprecated. Please don't use it")
    virtual const std::string& legacy_name() const = 0;

    virtual size_t pointer_hash() const noexcept = 0;

protected:
    virtual ~ITensorDescriptor();
};

/** @brief The TensorExtension defines developer API for ov::descriptor::Tensor. */
struct OPENVINO_API TensorExtension {
    /**
     * @brief Get the tensor descriptor object
     *
     * @param tensor Tensor descriptor to access its implementation.
     * @return Reference to Tensor description implementation.
     */
    static const ITensorDescriptor& get_descriptor(const Tensor& tensor);
    static std::shared_ptr<ITensorDescriptor>& get_descriptor_ptr(Tensor& tensor);

    /**
     * @brief The hasher of shared pointer Tensor descriptor.
     */
    struct OPENVINO_API Hasher {
        size_t operator()(const std::shared_ptr<Tensor>& tensor) const;
    };

    /**
     * @brief The comparator of shared pointer Tensor descriptor.
     */
    struct OPENVINO_API Equal {
        bool operator()(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) const;
    };
};
}  // namespace descriptor
}  // namespace ov
