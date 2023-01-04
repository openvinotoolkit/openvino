// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/attribute_visitor.hpp"


namespace ngraph {
namespace snippets {
class TensorDescriptorAttribute;
class TensorDescriptor {
    friend class TensorDescriptorAttribute;
public:
explicit TensorDescriptor(const Output<ov::Node>& node,
                              std::vector<size_t> subtensor_shape = {},
                              std::vector<size_t> layout = {});
explicit TensorDescriptor(const Output<const ov::Node>& node,
                              std::vector<size_t> subtensor_shape = {},
                              std::vector<size_t> layout = {});
    TensorDescriptor(std::vector<size_t> tensor_shape,
                     std::vector<size_t> subtensor_shape,
                     std::vector<size_t> layout = {});
    TensorDescriptor() = default;
    static TensorDescriptor deserialize(const std::string& serialized_info);
    std::string  serialize() const;
    std::vector<size_t> get_tensor() const {return m_tensor_shape;}
    std::vector<size_t> get_subtensor() const {return m_subtensor_shape;}
    std::vector<size_t> get_layout() const {return m_layout;}
    bool empty() const { return m_tensor_shape.empty() && m_layout.empty() && m_subtensor_shape.empty();}
    friend bool operator==(const TensorDescriptor& lhs, const TensorDescriptor& rhs);
    friend bool operator!=(const TensorDescriptor& lhs, const TensorDescriptor& rhs) {return !(lhs == rhs);}

private:
    void validate_arguments();
    /// \brief Original tensor shape
    std::vector<size_t> m_tensor_shape{};
    /// \brief Order of dimensions: NCHW == {0, 1, 2, 3}, NHWC == {0, 2, 3, 1}, NCHW16c == {0, 1, 2, 3, 1}
    std::vector<size_t> m_layout{};
    /// \brief Minimal tensor size that could be processed in one call
    std::vector<size_t> m_subtensor_shape{};
};

std::ostream& operator << (std::ostream&, const TensorDescriptor& td);
using TensorDescriptorPtr = std::shared_ptr<TensorDescriptor>;
class TensorDescriptorPtrVectorAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("TensorDescriptorVectorAttribute", "0");

    TensorDescriptorPtrVectorAttribute() = default;
    explicit TensorDescriptorPtrVectorAttribute(std::vector<TensorDescriptorPtr> descriptor) : m_value(std::move(descriptor)) {}
    std::vector<TensorDescriptorPtr> m_value{};
};

void set_tensor_descriptor_ptr(const Output<ov::Node>& n, const TensorDescriptorPtr& desc);
TensorDescriptorPtr get_tensor_descriptor_ptr(const Output<ov::Node>& out);
TensorDescriptorPtr get_tensor_descriptor_ptr(const Output<const ov::Node>& out);

} // namespace snippets
} // namespace ngraph
namespace ov {
using ngraph::snippets::TensorDescriptor;
template <>
class OPENVINO_API AttributeAdapter<TensorDescriptor> : public ov::ValueAccessor<std::string> {
public:
    OPENVINO_RTTI("AttributeAdapter<TensorDescriptor>");
    explicit AttributeAdapter(TensorDescriptor& value) : m_ref(value) {}

    const std::string& get() override;
    void set(const std::string& value) override;
    explicit operator TensorDescriptor&() {
        return m_ref;
    }

protected:
    TensorDescriptor& m_ref;
    std::string m_dump;
};
} // namespace ov
