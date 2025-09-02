// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <openvino/frontend/graph_iterator.hpp>

#include "graph_iterator_proto.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/wstring_convert_util.hpp"

namespace ov {
namespace frontend {
namespace onnx {

const std::string empty_name = "";
const std::string DEFAULT_DOMAIN = "";
const std::string EMPTY_NAME = "";
const std::string EMPTY_OP_TYPE = "";

ov::Any DecoderProto::get_attribute(const std::string& name) const {
    for (const auto& attr : m_node->attribute()) {
        if (!attr.has_name() || attr.name() != name)
            continue;
        if (!attr.has_type()) {
            throw std::runtime_error("Attribute \"" + name + "\" doesn't have a type");
        }
        switch (attr.type()) {
        case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT:
            if (attr.has_f())
                return attr.f();
            else
                throw std::runtime_error("Attribute doesn't have value");
            break;
        case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS:
            return std::vector<float>{attr.floats().begin(), attr.floats().end()};
        case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
            if (attr.has_i())
                return attr.i();
            else
                throw std::runtime_error("Attribute doesn't have value");
            break;
        case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS:
            return std::vector<int64_t>{attr.ints().begin(), attr.ints().end()};
        case AttributeProto_AttributeType::AttributeProto_AttributeType_STRING:
            if (attr.has_s())
                return attr.s();
            else
                throw std::runtime_error("Attribute doesn't have value");
            break;
        case AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS:
            return std::vector<std::string>{attr.strings().begin(), attr.strings().end()};
        case AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH:
            if (attr.has_g())
                return static_cast<ov::frontend::onnx::GraphIterator::Ptr>(
                    std::make_shared<GraphIteratorProto>(m_parent, &attr.g()));
            else
                throw std::runtime_error("Attribute doesn't have value");
            break;
        case AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR:
            return static_cast<ov::frontend::onnx::DecoderBase::Ptr>(
                std::make_shared<DecoderProtoTensor>(&attr.t(), m_parent, 0, 0));
        default:
            throw std::runtime_error("Unsupported attribute type " +
                                     ::ONNX_NAMESPACE::AttributeProto_AttributeType_Name(attr.type()));
        }
    }
    return nullptr;
}

size_t DecoderProto::get_input_size() const {
    return m_input_info.size();
}

size_t DecoderProto::get_output_size() const {
    return m_output_info.size();
}

void DecoderProto::get_input_node(size_t input_port_idx,
                                  std::string& producer_name,
                                  std::string& producer_output_port_name,
                                  size_t& producer_output_port_index) const {}

const std::string& DecoderProto::get_op_type() const {
    if (m_node->has_op_type()) {
        return m_node->op_type();
    } else {
        return EMPTY_OP_TYPE;
    }
}

const std::string& DecoderProto::get_op_name() const {
    if (m_node->has_name()) {
        return m_node->name();
    } else {
        return EMPTY_NAME;
    }
}

bool DecoderProto::has_attribute(const std::string& name) const {
    for (const auto& attr : m_node->attribute()) {
        if (attr.has_name() && attr.name() == name) {
            return true;
        }
    }
    return false;
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
