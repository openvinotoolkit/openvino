// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_proto.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <openvino/frontend/graph_iterator.hpp>

#include "decoder_proto.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/wstring_convert_util.hpp"

namespace {
// THis is copied from utils/common.hpp
const ov::element::Type& get_ov_element_type(int64_t onnx_type) {
    switch (onnx_type) {
    case TensorProto_DataType::TensorProto_DataType_BOOL:
        return ov::element::boolean;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
        return ov::element::f64;
    case TensorProto_DataType::TensorProto_DataType_FLOAT16:
        return ov::element::f16;
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
        return ov::element::f32;
    case TensorProto_DataType::TensorProto_DataType_INT4:
        return ov::element::i4;
    case TensorProto_DataType::TensorProto_DataType_INT8:
        return ov::element::i8;
    case TensorProto_DataType::TensorProto_DataType_INT16:
        return ov::element::i16;
    case TensorProto_DataType::TensorProto_DataType_INT32:
        return ov::element::i32;
    case TensorProto_DataType::TensorProto_DataType_INT64:
        return ov::element::i64;
    case TensorProto_DataType::TensorProto_DataType_UINT4:
        return ov::element::u4;
    case TensorProto_DataType::TensorProto_DataType_UINT8:
        return ov::element::u8;
    case TensorProto_DataType::TensorProto_DataType_UINT16:
        return ov::element::u16;
    case TensorProto_DataType::TensorProto_DataType_UINT32:
        return ov::element::u32;
    case TensorProto_DataType::TensorProto_DataType_UINT64:
        return ov::element::u64;
    case TensorProto_DataType::TensorProto_DataType_UNDEFINED:
        return ov::element::dynamic;
    case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
        return ov::element::bf16;
    case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
        return ov::element::f8e4m3;
    case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
        return ov::element::f8e5m2;
    case TensorProto_DataType::TensorProto_DataType_STRING:
        return ov::element::string;
    }
    throw std::runtime_error("Unsupported type");
}
}  // namespace

namespace ov {
namespace frontend {
namespace onnx {

ov::frontend::onnx::TensorMetaInfo extract_tensor_meta_info(const TensorProto* tensor_info,
                                                            const ValueInfoProto* value_info,
                                                            const GraphProto* graph_def) {
    ov::frontend::onnx::TensorMetaInfo tensor_meta_info{};
    if ((tensor_info == nullptr && value_info == nullptr) || graph_def == nullptr) {
        throw std::runtime_error("Wrong usage");
    }

    if (value_info == nullptr && tensor_info->has_name()) {
        for (const auto& val : graph_def->value_info()) {
            if (val.has_name() && val.name() == tensor_info->name()) {
                value_info = &val;
                break;
            }
        }
    }
    if (value_info != nullptr && value_info->has_type()) {
        if (!value_info->type().has_tensor_type()) {
            throw std::runtime_error("Unsupported value_info type");
        }
        tensor_meta_info.m_tensor_name = value_info->has_name() ? &value_info->name() : &empty_name;
        const auto& value_type = value_info->type().tensor_type();
        if (value_type.has_shape()) {
            std::vector<int64_t> dims{};
            for (const auto& dim : value_type.shape().dim()) {
                if (dim.has_dim_value()) {
                    dims.push_back(dim.dim_value());
                } else {
                    dims.push_back(-1);
                }
            }
            tensor_meta_info.m_partial_shape = ov::PartialShape{dims};
        } else {
            tensor_meta_info.m_partial_shape = ov::PartialShape::dynamic();
        }
        if (value_type.has_elem_type()) {
            tensor_meta_info.m_element_type = get_ov_element_type(value_type.elem_type());
        } else {
            tensor_meta_info.m_element_type = ov::element::dynamic;
        }
    }
    if (tensor_info != nullptr) {
        tensor_meta_info.m_tensor_name = tensor_info->has_name() ? &tensor_info->name() : &empty_name;
        tensor_meta_info.m_partial_shape =
            ov::PartialShape{std::vector<int64_t>{tensor_info->dims().begin(), tensor_info->dims().end()}};
        tensor_meta_info.m_element_type =
            tensor_info->has_data_type() ? get_ov_element_type(tensor_info->data_type()) : ov::element::dynamic;
        if (tensor_info->has_data_location() &&
            tensor_info->data_location() == TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
            throw std::runtime_error("Unexpected usage of method for externally stored data");
        }
        switch (tensor_info->data_type()) {
        case TensorProto_DataType::TensorProto_DataType_FLOAT:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->float_data().data()));
            tensor_meta_info.m_tensor_data_size = tensor_info->float_data_size();
            break;
        case TensorProto_DataType::TensorProto_DataType_INT4:
        case TensorProto_DataType::TensorProto_DataType_INT8:
        case TensorProto_DataType::TensorProto_DataType_INT16:
        case TensorProto_DataType::TensorProto_DataType_INT32:
        case TensorProto_DataType::TensorProto_DataType_UINT4:
        case TensorProto_DataType::TensorProto_DataType_UINT8:
        case TensorProto_DataType::TensorProto_DataType_UINT16:
        case TensorProto_DataType::TensorProto_DataType_UINT32:
        case TensorProto_DataType::TensorProto_DataType_BOOL:
        case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
        case TensorProto_DataType::TensorProto_DataType_FLOAT16:
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->int32_data().data()));
            tensor_meta_info.m_tensor_data_size = tensor_info->int32_data_size();
            break;
        case TensorProto_DataType::TensorProto_DataType_INT64:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->int64_data().data()));
            tensor_meta_info.m_tensor_data_size = tensor_info->int64_data_size();
            break;
        case TensorProto_DataType::TensorProto_DataType_UINT64:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->uint64_data().data()));
            tensor_meta_info.m_tensor_data_size = tensor_info->uint64_data_size();
            break;
        case TensorProto_DataType::TensorProto_DataType_DOUBLE:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->double_data().data()));
            tensor_meta_info.m_tensor_data_size = tensor_info->double_data_size();
            break;
        default:
            std::cout << ::ONNX_NAMESPACE::TensorProto_DataType_Name(tensor_info->data_type());
            throw std::runtime_error("Unsupported type " +
                                     ::ONNX_NAMESPACE::TensorProto_DataType_Name(tensor_info->data_type()));
            break;
        }
        // Looks like raw_data has bigger priority. but not 100% sure
        if (tensor_meta_info.m_tensor_data == nullptr && tensor_info->has_raw_data()) {
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->raw_data().data()));
            tensor_meta_info.m_tensor_data_size = tensor_info->raw_data().size();
        }
    }
    return tensor_meta_info;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

GraphIteratorProto::GraphIteratorProto(const std::wstring& path)
    : GraphIteratorProto(ov::util::wstring_to_string(path)) {}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

GraphIteratorProto::GraphIteratorProto(const std::string& path) : m_parent(nullptr) {
    // @TODO: This is a really bad thing - resource allocation in a constructor
    // Need to think about usage init()/reset() as a base for resource allocation
    try {
        std::ifstream model_file(path, std::ios::binary | std::ios::in);
        FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", path);

        m_model = std::make_shared<ModelProto>();
        m_model->ParseFromIstream(&model_file);
        model_file.close();
        if (m_model->has_graph()) {
            m_graph = &m_model->graph();
        } else {
            m_graph = nullptr;
            return;
        }
        reset();
    } catch (...) {
        m_model.reset();
        m_graph = nullptr;
        node_index = 0;
        m_decoders.clear();
    }
}

GraphIteratorProto::GraphIteratorProto(GraphIteratorProto* parent, const GraphProto* graph_def) {
    m_parent = parent;
    m_model = parent->m_model;
    m_graph = graph_def;
    if (m_graph != nullptr) {
        try {
            reset();
        } catch (...) {
        }
    }
}

std::shared_ptr<DecoderProtoTensor> GraphIteratorProto::get_tensor(const std::string& name,
                                                                   GraphIteratorProto** owner) {
    if (m_tensors.count(name) == 0) {
        if (name == empty_name) {
            *owner = this;
            const auto& tensor_decoder = std::make_shared<DecoderProtoTensor>(empty_name, this, -1, -1);
            m_tensors[empty_name] = tensor_decoder;
            return tensor_decoder;
        }
        if (m_parent == nullptr) {
            throw std::runtime_error("Input tensor isn't found for node \"" + name + "\"");
        }
        return m_parent->get_tensor(name, owner);
    }
    *owner = this;
    return m_tensors[name];
}

void GraphIteratorProto::reset() {
    node_index = 0;
    if (m_decoders.size() > 0 || m_model == nullptr || m_graph == nullptr)
        return;
    m_decoders.reserve(m_graph->initializer_size() + m_graph->input_size() + m_graph->output_size() +
                       m_graph->node_size());
    for (const auto& value : m_graph->input()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, this, 0, -1);
        m_decoders.push_back(tensor);
        if (m_tensors.count(*tensor->get_tensor_info().m_tensor_name) > 0) {
            throw std::runtime_error("Tensor already exists \"" + *tensor->get_tensor_info().m_tensor_name + "\"");
        }
        m_tensors[*tensor->get_tensor_info().m_tensor_name] = tensor;
    }
    for (const auto& value : m_graph->output()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, this, -1, 0);
        m_decoders.push_back(tensor);
        if (m_tensors.count(*tensor->get_tensor_info().m_tensor_name) > 0) {
            throw std::runtime_error("Tensor already exists \"" + *tensor->get_tensor_info().m_tensor_name + "\"");
        }
        m_tensors[*tensor->get_tensor_info().m_tensor_name] = tensor;
    }
    for (const auto& initializer : m_graph->initializer()) {
        const auto& decoder =
            std::find_if(m_decoders.begin(),
                         m_decoders.end(),
                         [&initializer](const std::shared_ptr<ov::frontend::onnx::DecoderBase>& value) {
                             const auto& tensor = std::dynamic_pointer_cast<DecoderProtoTensor>(value);
                             if (tensor == nullptr)
                                 return false;
                             return initializer.name() == *tensor->get_tensor_info().m_tensor_name;
                         });
        if (decoder != m_decoders.end()) {
            *const_cast<ov::frontend::onnx::TensorMetaInfo*>(
                &std::dynamic_pointer_cast<DecoderProtoTensor>(*decoder)->get_tensor_info()) =
                extract_tensor_meta_info(&initializer, nullptr, m_graph);
            continue;
        }
        const auto tensor = std::make_shared<DecoderProtoTensor>(&initializer, this, -1, -1);
        m_tensors[*tensor->get_tensor_info().m_tensor_name] = tensor;
    }
    size_t top_index = 0;
    for (const auto& node : m_graph->node()) {
        std::vector<const ov::frontend::onnx::TensorMetaInfo*> input_tensors{};
        std::vector<const ov::frontend::onnx::TensorMetaInfo*> output_tensors{};
        input_tensors.reserve(node.input_size());
        output_tensors.reserve(node.output_size());
        GraphIteratorProto* tensor_owner = nullptr;
        for (const auto& name : node.input()) {
            auto decoder_proto_tensor = this->get_tensor(name, &tensor_owner);
            input_tensors.push_back(&decoder_proto_tensor->get_tensor_info());
            if (tensor_owner != this) {
                // Need to insert parent's decoders on top of decoders
                m_decoders.insert(m_decoders.begin() + top_index, decoder_proto_tensor);
                ++top_index;
            }
        }
        for (const auto& name : node.output()) {
            if (name != "") {
                const auto& found_tensor = m_tensors.find(name);
                if (found_tensor == m_tensors.end()) {
                    const auto& initializer = std::find_if(m_graph->initializer().begin(),
                                                           m_graph->initializer().end(),
                                                           [&name](const TensorProto& value) {
                                                               return value.has_name() && value.name() == name;
                                                           });
                    std::shared_ptr<DecoderProtoTensor> tensor{nullptr};
                    if (initializer != m_graph->initializer().end()) {
                        tensor = std::make_shared<DecoderProtoTensor>(&*initializer, this, -1, -1);
                    } else {
                        const auto& value_info = std::find_if(m_graph->value_info().begin(),
                                                              m_graph->value_info().end(),
                                                              [&name](const ValueInfoProto& value) {
                                                                  return value.has_name() && value.name() == name;
                                                              });
                        if (value_info != m_graph->value_info().end())
                            tensor = std::make_shared<DecoderProtoTensor>(&*value_info, this, -1, -1);
                    }
                    if (tensor == nullptr) {
                        tensor = std::make_shared<DecoderProtoTensor>(name, this, -1, -1);
                    }
                    m_decoders.push_back(tensor);
                    m_tensors[name] = tensor;
                    output_tensors.push_back(&tensor->get_tensor_info());
                } else {
                    output_tensors.push_back(&found_tensor->second->get_tensor_info());
                }
            } else {
                output_tensors.push_back(&this->get_tensor(empty_name, &tensor_owner)->get_tensor_info());
            }
        }
        const std::string& domain = node.has_domain() && node.domain() != "ai.onnx" ? node.domain() : DEFAULT_DOMAIN;
        int64_t opset = get_opset_version(domain);
        if (opset == -1) {
            throw std::runtime_error("Operation version isn't found");
        }
        auto decoder_node =
            std::make_shared<DecoderProto>(&node, static_cast<uint64_t>(opset), this, input_tensors, output_tensors);
        m_decoders.push_back(decoder_node);
    }
}

size_t GraphIteratorProto::get_subgraph_size() const {
    return 0;
}

std::shared_ptr<ov::frontend::onnx::GraphIterator> GraphIteratorProto::get_subgraph(size_t idx) const {
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
    /*
    FRONT_END_GENERAL_CHECK(0 == idx, "There is no subgraph with idx ", idx);
    auto iterator = std::make_shared<GraphIteratorProto>();
    iterator->node_index = 0;
    iterator->m_model = m_model;
    iterator->m_subgraphs = {};  // TODO: check if we need to pass all sub-graphs here (while in a while situation)
    iterator->m_graph = m_subgraphs[idx];
    const auto operators = iterator->m_graph->operators();
    auto operators_vec = std::vector<const tflite::Operator*>{operators->begin(), operators->end()};
    iterator->m_nodes.assign(operators_vec.begin(), operators_vec.end());
    auto outputs = iterator->m_graph->outputs();
    auto inputs = iterator->m_graph->inputs();
    iterator->m_nodes.insert(iterator->m_nodes.begin(), outputs->begin(), outputs->end());
    iterator->m_nodes.insert(iterator->m_nodes.begin(), inputs->begin(), inputs->end());
    return iterator;
    */
}

std::shared_ptr<ov::frontend::onnx::DecoderBase> GraphIteratorProto::get_decoder() const {
    return m_decoders[node_index];
}

std::int64_t GraphIteratorProto::get_opset_version(const std::string& domain) const {
    // copy the opsets and sort them (descending order)
    // then return the version from the first occurrence of a given domain
    auto opset_imports = m_model->opset_import();
    std::sort(std::begin(opset_imports),
              std::end(opset_imports),
              [](const OperatorSetIdProto& lhs, const OperatorSetIdProto& rhs) {
                  return lhs.version() > rhs.version();
              });

    for (const auto& opset_import : opset_imports) {
        if (domain == opset_import.domain()) {
            return opset_import.version();
        }
    }

    return -1;
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
