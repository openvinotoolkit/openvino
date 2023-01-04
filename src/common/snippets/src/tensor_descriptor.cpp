// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/tensor_descriptor.hpp"
#include "ngraph/except.hpp"
#include <sstream>

namespace ngraph {
namespace snippets {
TensorDescriptor::TensorDescriptor(const Output<ov::Node>& out,
                                   std::vector<size_t> subtensor_shape,
                                   std::vector<size_t> layout)
                                   : TensorDescriptor(ov::Output<const Node>(out.get_node(), out.get_index()),
                                                      std::move(subtensor_shape),
                                                      std::move(layout)) {
}

TensorDescriptor::TensorDescriptor(const Output<const ov::Node>& out,
                                   std::vector<size_t> subtensor_shape,
                                   std::vector<size_t> layout)
        : m_subtensor_shape(std::move(subtensor_shape)), m_layout(std::move(layout)) {
    const auto& pshape = out.get_partial_shape();
    // Note: this limitation could be relaxed if necessary
    if (pshape.is_dynamic())
        throw ngraph_error("Snippets tensor descriptor can be created only for static shapes");
    m_tensor_shape = pshape.get_shape();
    validate_arguments();
}

TensorDescriptor::TensorDescriptor(std::vector<size_t> tensor_shape,
                                   std::vector<size_t> subtensor_shape,
                                   std::vector<size_t> layout) : m_tensor_shape(std::move(tensor_shape)),
                                   m_layout(std::move(layout)), m_subtensor_shape(std::move(subtensor_shape)) {
    validate_arguments();
}

void TensorDescriptor::validate_arguments() {
    if (!m_tensor_shape.empty() && m_layout.empty()) {
        m_layout.resize(m_tensor_shape.size());
        // NCHW layout by default
        std::iota(m_layout.begin(), m_layout.end(), 0);
    } else if (m_layout.size() != m_tensor_shape.size()) {
        throw ngraph_error("Snippets tensor descriptor: Layout size must be equal to the shape size");
    }
}


TensorDescriptor TensorDescriptor::deserialize(const std::string& serialized_info) {
    std::stringstream sinfo(serialized_info);
    auto read_values = [](std::stringstream& ss){
        size_t num = 0;
        ss >> num;
        std::vector<size_t> res;
        for (size_t i = 0; i < num; i++) {
            size_t val;
            ss >> val;
            res.push_back(val);
        }
        return res;
    };
    const auto& tensor_shape = read_values(sinfo);
    const auto& subtensor_shape = read_values(sinfo);
    const auto& layout = read_values(sinfo);
    return {tensor_shape, subtensor_shape, layout};
}

std::string  TensorDescriptor::serialize() const {
    std::stringstream ss;
    ss << m_tensor_shape.size() << " ";
    for (auto val : m_tensor_shape)
        ss << val << " ";
    ss << m_subtensor_shape.size() << " ";
    for (auto val : m_subtensor_shape)
        ss << val << " ";
    ss << m_layout.size() << " ";
    for (auto val : m_layout)
        ss << val << " ";
    return ss.str();
}
bool operator==(const TensorDescriptor& lhs, const TensorDescriptor& rhs) {
    return lhs.m_tensor_shape == rhs.m_tensor_shape &&
           lhs.m_layout == rhs.m_layout &&
           lhs.m_subtensor_shape == rhs.m_subtensor_shape;
}

std::ostream& operator << (std::ostream& ss, const TensorDescriptor& td) {
    auto print_vector = [&ss](const std::vector<size_t>& data){
        ss << "[";
        for (auto i : data)
            ss << i << ",";
        ss << (data.empty() ? "]" : "\b]");
    };
    ss  << "{Tensor: ";
    print_vector(td.get_tensor());
    ss  << " Subtensor: ";
    print_vector(td.get_subtensor());
    ss  << " Layout: ";
    print_vector(td.get_layout());
    ss << "}";
    return ss;
}

void set_tensor_descriptor_ptr(const Output<ov::Node>& out, const TensorDescriptorPtr& desc) {
    const auto& node = out.get_node_shared_ptr();
    auto& rt_info = node->get_rt_info();
    const auto& key = TensorDescriptorPtrVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found  == rt_info.end()) {
        std::vector<TensorDescriptorPtr> value(node->get_output_size());
        value[out.get_index()] = desc;
        rt_info[key] = TensorDescriptorPtrVectorAttribute(value);
    } else {
        auto& value = found->second.as<TensorDescriptorPtrVectorAttribute>().m_value;
        if (value.size() != node->get_output_size())
            throw ngraph_error("Either all or none of Tensor descriptors should be stored in rt_info (set)");
        value[out.get_index()] = desc;
    }
}
TensorDescriptorPtr get_tensor_descriptor_ptr(const Output<ov::Node>& out) {
    return get_tensor_descriptor_ptr(ov::Output<const Node>(out.get_node(), out.get_index()));
}
TensorDescriptorPtr get_tensor_descriptor_ptr(const Output<const ov::Node>& out) {
    const auto& node = out.get_node_shared_ptr();
    const auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(TensorDescriptorPtrVectorAttribute::get_type_info_static());
    if (it == rt_info.end()) {
        return std::make_shared<TensorDescriptor>(out);
    }
    const auto& td_vector = it->second.as<TensorDescriptorPtrVectorAttribute>().m_value;
    if (td_vector.size() != node->get_output_size())
        throw ngraph_error("Either all or none of Tensor descriptors should be stored in rt_info (get)");
    return td_vector[out.get_index()];
}
} // namespace snippets
} // namespace ngraph
namespace ov {
const std::string& ov::AttributeAdapter<TensorDescriptor>::get() {
    m_dump = m_ref.serialize();
    return m_dump;
}

void ov::AttributeAdapter<TensorDescriptor>::set(const std::string& value) {
    m_ref = TensorDescriptor::deserialize(value);
}
} // namespace ov
