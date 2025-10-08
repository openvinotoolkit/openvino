// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <onnx/onnx_pb.h>

#include <openvino/frontend/graph_iterator.hpp>

#include "graph_iterator_proto.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/wstring_convert_util.hpp"

using ::ONNX_NAMESPACE::AttributeProto_AttributeType;
using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::OperatorSetIdProto;
using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::TensorProto_DataLocation;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ::ONNX_NAMESPACE::ValueInfoProto;
using ::ONNX_NAMESPACE::Version;

namespace ov {
namespace frontend {
namespace onnx {

ov::frontend::onnx::TensorMetaInfo extract_tensor_meta_info(const TensorProto* tensor_info,
                                                            const ValueInfoProto* value_info,
                                                            GraphIteratorProto* graph_iterator);

extern const std::string empty_name;
extern const std::string DEFAULT_DOMAIN;
extern const std::string EMPTY_NAME;
extern const std::string EMPTY_OP_TYPE;

class DecoderProtoTensor : public ov::frontend::onnx::DecoderBaseTensor {
    ov::frontend::onnx::TensorMetaInfo m_tensor_meta_info;
    int64_t m_input_idx, m_output_idx;

public:
    DecoderProtoTensor(const TensorProto* tensor_info,
                       GraphIteratorProto* parent,
                       const int64_t input_idx,
                       const int64_t output_idx)
        // Probably, we may need to force it to 0/0
        : m_input_idx(input_idx),
          m_output_idx(output_idx) {
        m_tensor_meta_info = extract_tensor_meta_info(tensor_info, nullptr, parent);
    }
    DecoderProtoTensor(const ValueInfoProto* value_info,
                       GraphIteratorProto* parent,
                       const int64_t input_idx,
                       const int64_t output_idx)
        : m_input_idx(input_idx),
          m_output_idx(output_idx) {
        m_tensor_meta_info = extract_tensor_meta_info(nullptr, value_info, parent);
    }
    DecoderProtoTensor(const std::string& name,
                       GraphIteratorProto* parent,
                       const int64_t input_idx,
                       const int64_t output_idx)
        : m_input_idx(input_idx),
          m_output_idx(output_idx) {
        m_tensor_meta_info.m_tensor_name = &name;
        m_tensor_meta_info.m_element_type = ov::element::dynamic;
        m_tensor_meta_info.m_partial_shape = ov::PartialShape::dynamic();
        m_tensor_meta_info.m_tensor_data = nullptr;
        m_tensor_meta_info.m_tensor_data_size = 0;
    }

    const ov::frontend::onnx::TensorMetaInfo& get_tensor_info() const override {
        return *const_cast<const ov::frontend::onnx::TensorMetaInfo*>(&m_tensor_meta_info);
    }

    int64_t get_input_idx() const override {
        return m_input_idx;
    }

    int64_t get_output_idx() const override {
        return m_output_idx;
    }

    ov::Any get_attribute(const std::string& name) const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_attribute");
    }

    size_t get_input_size() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_input_size");
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_input_node");
    }

    const std::string& get_op_type() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_op_type");
    }

    const std::string& get_op_name() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_op_name");
    }
};

class DecoderProto : public ov::frontend::onnx::DecoderBaseOperation {
    const NodeProto* m_node;
    uint64_t m_opset;
    GraphIteratorProto* m_parent;  // For future use
    // For existence of NodeDef object corresponding to the main graph node,
    // GraphDef object must live in the memory
    const GraphProto* m_graph;
    std::vector<const ov::frontend::onnx::TensorMetaInfo*> m_input_info, m_output_info;

public:
    explicit DecoderProto(const NodeProto* node_def,
                          const uint64_t opset,
                          GraphIteratorProto* parent,
                          const std::vector<const ov::frontend::onnx::TensorMetaInfo*>& input_info,
                          const std::vector<const ov::frontend::onnx::TensorMetaInfo*>& output_info)
        : m_node(node_def),
          m_opset(opset),
          m_parent(parent),
          m_graph(parent->get_graph()),
          m_input_info(input_info),
          m_output_info(output_info) {}

    size_t get_input_size() const override;
    size_t get_output_size() const override;

    const std::string& get_input_tensor_name(size_t idx) const override {
        return *m_input_info.at(idx)->m_tensor_name;
    }
    ov::element::Type get_input_tensor_type(size_t idx) const override {
        return m_input_info.at(idx)->m_element_type;
    }
    const std::string& get_output_tensor_name(size_t idx) const override {
        return *m_output_info.at(idx)->m_tensor_name;
    }
    ov::element::Type get_output_tensor_type(size_t idx) const override {
        return m_output_info.at(idx)->m_element_type;
    }
    const ov::frontend::onnx::TensorMetaInfo& get_input_tensor_info(size_t idx) const override {
        return *m_input_info.at(idx);
    }
    const ov::frontend::onnx::TensorMetaInfo& get_output_tensor_info(size_t idx) const override {
        return *m_output_info.at(idx);
    }

    ov::Any get_attribute(const std::string& name) const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

    uint64_t get_op_set() const override {
        return m_opset;
    }

    const std::string& get_domain() const override {
        return (m_node->has_domain() && m_node->domain() != "ai.onnx" ? m_node->domain() : DEFAULT_DOMAIN);
    }

    bool has_attribute(const std::string& name) const override;

    void experimental_get_internal_structures(const void** node_def) const override {
        *node_def = m_node;
    }
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
