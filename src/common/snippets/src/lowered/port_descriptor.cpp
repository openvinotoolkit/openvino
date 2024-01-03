// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/port_descriptor.hpp"
#include <snippets/utils.hpp>

namespace ov {
namespace snippets {
namespace lowered {

size_t PortDescriptor::ServiceDimensions::FULL_DIM = SIZE_MAX;

PortDescriptor::PortDescriptor(const ov::Input<ov::Node>& in, VectorDims subtensor_shape, std::vector<size_t> layout)
        : PortDescriptor(ov::Input<const Node>(in.get_node(), in.get_index()), std::move(subtensor_shape), std::move(layout)) {}

PortDescriptor::PortDescriptor(const ov::Input<const ov::Node>& in, std::vector<size_t> subtensor_shape, std::vector<size_t> layout)
        : PortDescriptor(utils::pshape_to_vdims(in.get_partial_shape()), std::move(subtensor_shape), std::move(layout)) {}

PortDescriptor::PortDescriptor(const ov::Output<ov::Node>& out, VectorDims subtensor_shape, std::vector<size_t> layout)
        : PortDescriptor(ov::Output<const Node>(out.get_node(), out.get_index()), std::move(subtensor_shape), std::move(layout)) {}

PortDescriptor::PortDescriptor(const ov::Output<const ov::Node>& out, std::vector<size_t> subtensor_shape, std::vector<size_t> layout)
        : PortDescriptor(utils::pshape_to_vdims(out.get_partial_shape()), std::move(subtensor_shape), std::move(layout)) {}

PortDescriptor::PortDescriptor(VectorDims shape, VectorDims subtensor_shape, std::vector<size_t> layout)
    : m_tensor_shape(std::move(shape)), m_layout(std::move(layout)), m_subtensor_shape(std::move(subtensor_shape)) {
    validate_arguments();
}

void PortDescriptor::validate_arguments() {
    if (!m_tensor_shape.empty() && m_layout.empty()) {
        m_layout.resize(m_tensor_shape.size());
        // NCHW layout by default
        std::iota(m_layout.begin(), m_layout.end(), 0);
    }
    OPENVINO_ASSERT(m_layout.size() == m_tensor_shape.size(), "Snippets tensor descriptor: Layout size must be equal to the shape size");
}

PortDescriptorPtr PortDescriptor::clone() const {
    auto desc = std::make_shared<PortDescriptor>(m_tensor_shape, m_subtensor_shape, m_layout);
    desc->set_reg(m_reg);
    return desc;
}

std::string  PortDescriptor::serialize() const {
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
bool operator==(const PortDescriptor& lhs, const PortDescriptor& rhs) {
    return lhs.m_tensor_shape == rhs.m_tensor_shape &&
           lhs.m_layout == rhs.m_layout &&
           lhs.m_subtensor_shape == rhs.m_subtensor_shape;
}

void PortDescriptorUtils::init_default(std::vector<PortDescriptorPtr>& in_descs,
                                       std::vector<PortDescriptorPtr>& out_descs,
                                       const std::shared_ptr<ov::Node>& node) {
    in_descs.resize(node->get_input_size());
    out_descs.resize(node->get_output_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        in_descs[i] = std::make_shared<PortDescriptor>(node->input(i));
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        out_descs[i] = std::make_shared<PortDescriptor>(node->output(i));
    }
}

void PortDescriptorUtils::set_port_descriptor_ptr(const ov::Input<ov::Node>& in, const PortDescriptorPtr& desc) {
    const auto& node = in.get_node()->shared_from_this();
    auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        std::vector<PortDescriptorPtr> in_descs, out_descs;
        init_default(in_descs, out_descs, node);
        in_descs[in.get_index()] = desc;
        rt_info[key] = PortDescriptorVectorAttribute(in_descs, out_descs);
    } else {
        auto& in_descs = found->second.as<PortDescriptorVectorAttribute>().inputs;
        if (in_descs.size() != node->get_input_size())
            OPENVINO_THROW("Set input port descriptor is failed: incorrect count");
        in_descs[in.get_index()] = desc;
    }
}

void PortDescriptorUtils::set_port_descriptor_ptr(const ov::Output<ov::Node>& out, const PortDescriptorPtr& desc) {
    const auto& node = out.get_node_shared_ptr();
    auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        std::vector<PortDescriptorPtr> in_descs, out_descs;
        init_default(in_descs, out_descs, node);
        out_descs[out.get_index()] = desc;
        rt_info[key] = PortDescriptorVectorAttribute(in_descs, out_descs);
    } else {
        auto& out_descs = found->second.as<PortDescriptorVectorAttribute>().outputs;
        if (out_descs.size() != node->get_output_size())
            OPENVINO_THROW("Set output port descriptor is failed: incorrect count");
        out_descs[out.get_index()] = desc;
    }
}

PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const ov::Input<ov::Node>& in) {
    return get_port_descriptor_ptr(ov::Input<const Node>(in.get_node(), in.get_index()));
}
PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const ov::Input<const ov::Node>& in) {
    const auto& node = in.get_node();
    auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        return std::make_shared<PortDescriptor>(in);
    }
    const auto& in_descs = found->second.as<PortDescriptorVectorAttribute>().inputs;
    if (in_descs.size() != node->get_input_size())
        OPENVINO_THROW("Get input port descriptor is failed: incorrect count");
    return in_descs[in.get_index()];
}

PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const Output<ov::Node>& out) {
    return get_port_descriptor_ptr(ov::Output<const Node>(out.get_node(), out.get_index()));
}
PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const Output<const ov::Node>& out) {
    const auto& node = out.get_node();
    const auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        return std::make_shared<PortDescriptor>(out);
    }
    const auto& out_descs = found->second.as<PortDescriptorVectorAttribute>().outputs;
    if (out_descs.size() != node->get_output_size())
        OPENVINO_THROW("Get output port descriptor is failed: incorrect count");
    return out_descs[out.get_index()];
}

void PortDescriptorUtils::clean(const std::shared_ptr<ov::Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(PortDescriptorVectorAttribute::get_type_info_static());
}
} // namespace lowered
} // namespace snippets
} // namespace ov
