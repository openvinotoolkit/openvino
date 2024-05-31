// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/runtime/tensor.hpp"

namespace ov {
class Node;
/// \brief Alias for symbol tensor.
using TensorSymbol = std::vector<std::shared_ptr<Symbol>>;
/// \brief Alias for vector of symbol tensors.

using TensorSymbolVector = std::vector<TensorSymbol>;

namespace pass {
class ReverseShapeAndTypeInfer;
}
namespace descriptor {

class Tensor;

/// \brief Compile-time descriptor of a first-class value that is a tensor.
class OPENVINO_API Tensor {
public:
    Tensor(const element::Type& element_type,
           const PartialShape& pshape,
           const std::unordered_set<std::string>& names = {});
    Tensor(const element::Type& element_type, const PartialShape& pshape, Node* node, size_t node_output_number);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    const std::string& get_any_name() const;
    const std::unordered_set<std::string>& get_names() const;
    void set_names(const std::unordered_set<std::string>& names);
    void add_names(const std::unordered_set<std::string>& names);

    /// \brief sets lower bound value description
    void set_lower_value(const ov::Tensor& value);
    /// \brief sets upper bound value description
    void set_upper_value(const ov::Tensor& value);
    /// \brief sets value symbol description
    void set_value_symbol(const TensorSymbol& value_symbol);
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
    const ov::Tensor& get_lower_value() const {
        return m_lower_value;
    }
    /// \brief gets upper bound value description
    const ov::Tensor& get_upper_value() const {
        return m_upper_value;
    }
    /// \brief gets symbol value description
    TensorSymbol get_value_symbol() const {
        return m_value_symbol;
    }
    /// \brief checks if lower and upper bound are set and point to the same Tensor
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

    PartialShape m_partial_shape;
    ov::Tensor m_lower_value, m_upper_value;
    TensorSymbol m_value_symbol;
    std::string m_legacy_name;

    std::unordered_set<std::string> m_names;
    std::unordered_set<std::string>::const_iterator m_name_it;
    RTMap m_rt_info;

    friend OPENVINO_API std::string get_ov_tensor_legacy_name(const Tensor& tensor);
    friend OPENVINO_API void set_ov_tensor_legacy_name(Tensor& tensor, const std::string& tensor_name);
    friend void set_element_type(Tensor& tensor, const element::Type& elemenet_type);
    friend void set_tensor_type(Tensor& tensor, const element::Type& element_type, const PartialShape& pshape);
    friend class pass::ReverseShapeAndTypeInfer;

private:
    mutable std::atomic<bool> m_shape_changing{false};
    mutable bool m_shape_changed{true};
    mutable Shape m_shape;
};

OPENVINO_API
std::ostream& operator<<(std::ostream&, const ov::descriptor::Tensor&);
}  // namespace descriptor
}  // namespace ov
