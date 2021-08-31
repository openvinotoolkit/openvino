// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include "ngraph/shape.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ngraph {
namespace runtime {
class HostTensor;
}
using HostTensorPtr = std::shared_ptr<runtime::HostTensor>;
}  // namespace ngraph

namespace ov {
class Node;
namespace descriptor {
/// \brief Compile-time descriptor of a first-class value that is a tensor.
class OPENVINO_API Tensor {
public:
    Tensor(const element::Type& element_type, const PartialShape& pshape, const std::string& name);
    Tensor(const element::Type& element_type, const PartialShape& pshape, Node* node, size_t node_output_number);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    NGRAPH_DEPRECATED("get_name() is deprecated! Please use get_names() instead.")
    const std::string& get_name() const;
    NGRAPH_DEPRECATED("set_name() is deprecated! Please use set_names() instead.")
    void set_name(const std::string& name);

    const std::unordered_set<std::string>& get_names() const;
    void set_names(const std::unordered_set<std::string>& names);
    void add_names(const std::unordered_set<std::string>& names);
    void set_tensor_type(const element::Type& element_type, const PartialShape& pshape);
    void set_element_type(const element::Type& elemenet_type);
    void set_partial_shape(const PartialShape& partial_shape);

    /// \brief sets lower bound value description
    void set_lower_value(const ngraph::HostTensorPtr& value);
    /// \brief sets upper bound value description
    void set_upper_value(const ngraph::HostTensorPtr& value);
    /// \brief unsets bound value descriptions
    void invalidate_values();

    const element::Type& get_element_type() const {
        return m_element_type;
    }
    const ngraph::Shape& get_shape() const;
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
    /// \brief checks if lower and upper bound are set and point to the same HostTensor
    bool has_and_set_bound() const {
        return m_upper_value != nullptr && m_upper_value == m_lower_value;
    }
    size_t size() const;

protected:
    element::Type m_element_type;

    // TODO: remove along with get_shape
    // Initially there was ngraph::Shape m_shape only available to keep shape information.
    // Support for dynamic shapes required transition to ngraph::PartialShape.
    // To smoothly transition to ngraph::PartialShape we introduced m_partial_shape
    // and kept m_shape in sync with m_partial_shape. Synchronization point was placed
    // in set_partial_shape which dramatically affected performance of ngraph::Function
    // validation. Since we have started the transition to ngraph::PartialShape and reduced
    // ngraph::Shape usage the only user of m_shape was get_shape method with signature:
    // const Shape& descriptor::Tensor::get_shape() const
    // It was decided to move m_shape and m_partial_shape synchronization point there and
    // to keep methods signature backward compatible.
    mutable std::mutex shape_mutex;
    mutable std::atomic_bool m_shape_changed;
    mutable ngraph::Shape m_shape;
    // TODO: end

    PartialShape m_partial_shape;
    ngraph::HostTensorPtr m_lower_value, m_upper_value;
    std::string m_name;
    std::unordered_set<std::string> m_names;
};

OPENVINO_API
std::ostream& operator<<(std::ostream&, const ov::descriptor::Tensor&);
}  // namespace descriptor
}  // namespace ov
