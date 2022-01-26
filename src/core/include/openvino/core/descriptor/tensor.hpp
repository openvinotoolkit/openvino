// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include "openvino/core/any.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ngraph {
namespace runtime {
class HostTensor;
}
using HostTensorPtr = std::shared_ptr<runtime::HostTensor>;
}  // namespace ngraph

namespace ov {
class Node;
using TensorLabel = std::vector<size_t>;
namespace descriptor {
/// \brief Compile-time descriptor of a first-class value that is a tensor.
class OPENVINO_API Tensor {
public:
    Tensor(const element::Type& element_type, const PartialShape& pshape, const std::string& name);
    Tensor(const element::Type& element_type, const PartialShape& pshape, Node* node, size_t node_output_number);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    OPENVINO_DEPRECATED("get_name() is deprecated! Please use get_names() instead.")
    const std::string& get_name() const;
    OPENVINO_DEPRECATED("set_name() is deprecated! Please use set_names() instead.")
    void set_name(const std::string& name);

    std::string get_any_name() const;
    const std::unordered_set<std::string>& get_names() const;
    void set_names(const std::unordered_set<std::string>& names);
    void add_names(const std::unordered_set<std::string>& names);

    OPENVINO_DEPRECATED("set_tensor_type() is deprecated. To change Tensor type please change the Parameter type")
    void set_tensor_type(const element::Type& element_type, const PartialShape& pshape);
    OPENVINO_DEPRECATED(
        "set_element_type() is deprecated. To change Tensor element type please change the Parameter type")
    void set_element_type(const element::Type& elemenet_type);
    OPENVINO_DEPRECATED(
        "set_partial_shape() is deprecated. To change Tensor partial shape please change the Parameter partial shape")
    void set_partial_shape(const PartialShape& partial_shape);

    /// \brief sets lower bound value description
    void set_lower_value(const ngraph::HostTensorPtr& value);
    /// \brief sets upper bound value description
    void set_upper_value(const ngraph::HostTensorPtr& value);
    /// \brief sets value label description
    void set_value_label(const TensorLabel& value_label);
    /// \brief unsets bound value descriptions
    void invalidate_values();

    const element::Type& get_element_type() const {
        return m_element_type;
    }
    const Shape& get_shape() const;
    const PartialShape& get_partial_shape() const {
        return m_partial_shape;
    }
    /// \brief gets lower bound value description
    ngraph::HostTensorPtr get_lower_value() const {
        return m_lower_value;
    }
    /// \brief gets upper bound value description
    ngraph::HostTensorPtr get_upper_value() const {
        return m_upper_value;
    }
    /// \brief gets upper bound value description
    TensorLabel get_value_label() const {
        return m_value_label;
    }
    /// \brief checks if lower and upper bound are set and point to the same HostTensor
    bool has_and_set_bound() const {
        return m_upper_value != nullptr && m_upper_value == m_lower_value;
    }
    size_t size() const;

    RTMap& get_rt_info() {
        return m_rt_info;
    }
    const RTMap& get_rt_info() const {
        return m_rt_info;
    }

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
    ngraph::HostTensorPtr m_lower_value, m_upper_value;
    TensorLabel m_value_label;
    std::string m_name;

    std::unordered_set<std::string> m_names;
    RTMap m_rt_info;
    mutable std::atomic_bool m_shape_changed;
};

OPENVINO_API
std::ostream& operator<<(std::ostream&, const ov::descriptor::Tensor&);
}  // namespace descriptor
}  // namespace ov
