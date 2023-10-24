// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/core/deprecated.hpp>
#include <string>
#include <unordered_set>

#include "openvino/core/any.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ngraph {
namespace runtime {
class HostTensor;
}
}  // namespace ngraph

namespace ov {
class Node;
/// \brief Alias for label tensor.
using TensorLabel = std::vector<label_t>;
/// \brief Alias for vector of label tensors.
using TensorLabelVector = std::vector<TensorLabel>;

namespace pass {
class ReverseShapeAndTypeInfer;
}
namespace descriptor {

class Tensor;

OPENVINO_DEPRECATED("get_ov_tensor_legacy_name() is deprecated. Please don't use this function.")
OPENVINO_API
std::string get_ov_tensor_legacy_name(const Tensor& tensor);

OPENVINO_DEPRECATED("set_ov_tensor_legacy_name() is deprecated. Please don't use this function.")
OPENVINO_API
void set_ov_tensor_legacy_name(Tensor& tensor, const std::string& tensor_name);

/// \brief Compile-time descriptor of a first-class value that is a tensor.
class OPENVINO_API Tensor {
public:
    Tensor(const element::Type& element_type,
           const PartialShape& pshape,
           const std::unordered_set<std::string>& names = {});
    OPENVINO_DEPRECATED("This constructor is deprecated. Please use constructor with set of names")
    Tensor(const element::Type& element_type, const PartialShape& pshape, const std::string& name);
    Tensor(const element::Type& element_type, const PartialShape& pshape, Node* node, size_t node_output_number);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    const std::string& get_any_name() const;
    const std::unordered_set<std::string>& get_names() const;
    void set_names(const std::unordered_set<std::string>& names);
    void add_names(const std::unordered_set<std::string>& names);

    OPENVINO_DEPRECATED("set_tensor_type() is deprecated. To change Tensor type please change the Parameter type")
    void set_tensor_type(const element::Type& element_type, const PartialShape& pshape);
    OPENVINO_DEPRECATED(
        "set_element_type() is deprecated. To change Tensor element type please change the Parameter type")
    void set_element_type(const element::Type& elemenet_type);

    /// \brief sets lower bound value description
    void set_lower_value(const ov::Tensor& value);
    /// \brief sets upper bound value description
    void set_upper_value(const ov::Tensor& value);
    /// \brief sets value label description
    void set_value_label(const TensorLabel& value_label);
    /// \brief unsets bound value descriptions
    void invalidate_values();

    const element::Type& get_element_type() const {
        return m_element_type;
    }
    OPENVINO_DEPRECATED("This method is deprecated and will be removed in 2024.0 release. Please use "
                        "get_partial_shape() method instead.")
    const Shape& get_shape() const;
    const PartialShape& get_partial_shape() const {
        return m_partial_shape;
    }
    /// \brief gets lower bound value description
    const ov::Tensor& get_lower_value() const {
        return m_lower_value;
    }
    /// \brief gets upper bound value description
    const ov::Tensor& get_upper_value() const {
        return m_upper_value;
    }
    /// \brief gets upper bound value description
    TensorLabel get_value_label() const {
        return m_value_label;
    }
    /// \brief checks if lower and upper bound are set and point to the same HostTensor
    bool has_and_set_bound() const {
        return m_upper_value && m_lower_value && m_upper_value.data() == m_lower_value.data();
    }
    size_t size() const;

    RTMap& get_rt_info() {
        return m_rt_info;
    }
    const RTMap& get_rt_info() const {
        return m_rt_info;
    }

    void clone_from(const Tensor& old);

protected:
    element::Type m_element_type;

    // TODO: remove along with get_shape
    // Initially there was Shape m_shape only available to keep shape information.
    // Support for dynamic shapes required transition to ov::PartialShape.
    // To smoothly transition to ov::PartialShape we introduced m_partial_shape
    // and kept m_shape in sync with m_partial_shape. Synchronization point was placed
    // in set_partial_shape which dramatically affected performance of ov::Model
    // validation. Since we have started the transition to ov::PartialShape and reduced
    // Shape usage the only user of m_shape was get_shape method with signature:
    // const PartialShape& descriptor::Tensor::get_shape() const
    // It was decided to move m_shape and m_partial_shape synchronization point there and
    // to keep methods signature backward compatible.
    mutable std::mutex m_mutex;
    mutable Shape m_shape;
    // TODO: end

    PartialShape m_partial_shape;
    ov::Tensor m_lower_value, m_upper_value;
    TensorLabel m_value_label;
    std::string m_legacy_name;

    std::unordered_set<std::string> m_names;
    std::unordered_set<std::string>::const_iterator m_name_it;
    RTMap m_rt_info;
    mutable std::atomic_bool m_shape_changed;

    friend OPENVINO_API std::string get_ov_tensor_legacy_name(const Tensor& tensor);
    friend OPENVINO_API void set_ov_tensor_legacy_name(Tensor& tensor, const std::string& tensor_name);
    friend class pass::ReverseShapeAndTypeInfer;
    friend class ngraph::runtime::HostTensor;
};

OPENVINO_API
std::ostream& operator<<(std::ostream&, const ov::descriptor::Tensor&);
}  // namespace descriptor
}  // namespace ov
