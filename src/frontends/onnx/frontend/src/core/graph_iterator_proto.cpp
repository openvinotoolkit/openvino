// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_proto.hpp"

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <map>
#include <unordered_set>
#include <vector>

#include "decoder_proto.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/frontend/graph_iterator.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/wstring_convert_util.hpp"
#include "utils/tensor_external_data.hpp"

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

namespace {
bool extract_tensor_external_data(ov::frontend::onnx::TensorMetaInfo& tensor_meta_info,
                                  const TensorProto* tensor_info,
                                  GraphIteratorProto* graph_iterator) {
    const auto ext_data = detail::TensorExternalData(*tensor_info);
    if (ext_data.data_location() == detail::ORT_MEM_ADDR) {
        tensor_meta_info.m_buffer = ext_data.load_external_mem_data();
    } else if (graph_iterator->get_mmap_cache()) {
        tensor_meta_info.m_buffer =
            ext_data.load_external_mmap_data(graph_iterator->get_model_dir(), graph_iterator->get_mmap_cache());
    } else {
        tensor_meta_info.m_buffer = ext_data.load_external_data(graph_iterator->get_model_dir());
    }
    return tensor_meta_info.m_buffer != nullptr;
}

template <typename T, typename Container>
std::shared_ptr<ov::AlignedBuffer> make_buffer_from_container_using_cast(const Container& container) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(container.size() * sizeof(T));
    T* ptr = buffer->template get_ptr<T>();
    size_t idx = 0;
    for (const auto& elem : container) {
        ptr[idx++] = static_cast<T>(elem);
    }
    return buffer;
}

template <typename T, typename Container>
std::shared_ptr<ov::AlignedBuffer> make_buffer_from_container(const Container& container) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(container.size() * sizeof(T));
    std::copy(container.begin(), container.end(), buffer->template get_ptr<T>());
    return buffer;
}
}  // namespace

ov::frontend::onnx::TensorMetaInfo extract_tensor_meta_info(const TensorProto* tensor_info,
                                                            const ValueInfoProto* value_info,
                                                            GraphIteratorProto* graph_iterator) {
    auto graph_def = graph_iterator->get_graph();
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
    if (value_info != nullptr) {
        tensor_meta_info.m_tensor_name = value_info->has_name() ? &value_info->name() : &empty_name;
        if (value_info->has_type() && !value_info->type().has_tensor_type()) {
            throw std::runtime_error("Unsupported value_info type: " + (*tensor_meta_info.m_tensor_name));
        }
        const auto& value_type = value_info->type().tensor_type();
        if (value_type.has_shape()) {
            std::vector<int64_t> dims;
            for (const auto& dim : value_type.shape().dim()) {
                dims.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
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
    int tensor_size = 0;
    if (tensor_info != nullptr) {
        tensor_meta_info.m_tensor_name = tensor_info->has_name() ? &tensor_info->name() : &empty_name;
        std::vector<int64_t> dims_vec{tensor_info->dims().begin(), tensor_info->dims().end()};
        tensor_meta_info.m_partial_shape = ov::PartialShape(dims_vec);
        tensor_meta_info.m_element_type =
            tensor_info->has_data_type() ? get_ov_element_type(tensor_info->data_type()) : ov::element::dynamic;
        if (tensor_info->has_data_location() &&
            tensor_info->data_location() == TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
            if (extract_tensor_external_data(tensor_meta_info, tensor_info, graph_iterator)) {
                auto element_count = tensor_meta_info.m_buffer->size() / tensor_meta_info.m_element_type.size();
                if (ov::element::is_nibble_type(tensor_meta_info.m_element_type)) {
                    element_count *= 2;  // Each byte contains 2 data items, so byte size must be multiplied
                }
                if (element_count != ov::shape_size(tensor_meta_info.m_partial_shape.get_shape())) {
                    FRONT_END_THROW(
                        "The size of the external data file does not match the byte size of an initializer '" +
                        *tensor_meta_info.m_tensor_name + "' in the model");
                }
                return tensor_meta_info;
            }
            throw std::runtime_error("Unsupported method for externally stored data");
        }

        if (tensor_info->has_segment()) {
            FRONT_END_THROW("Loading segments isn't supported");
        } else if (tensor_info->has_raw_data()) {
            tensor_size =
                static_cast<int>(tensor_info->raw_data().size() * 8 / tensor_meta_info.m_element_type.bitwidth());
            tensor_meta_info.m_buffer = std::make_shared<ov::AlignedBuffer>(tensor_info->raw_data().size());
            std::copy(tensor_info->raw_data().begin(),
                      tensor_info->raw_data().end(),
                      tensor_meta_info.m_buffer->get_ptr<char>());
        } else {
            switch (tensor_info->data_type()) {
            case TensorProto_DataType::TensorProto_DataType_INT32:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container<int32_t>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_INT4:
            case TensorProto_DataType::TensorProto_DataType_INT8:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container_using_cast<int8_t>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_INT16:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container_using_cast<int16_t>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_UINT4:
            case TensorProto_DataType::TensorProto_DataType_UINT8:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container_using_cast<uint8_t>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_UINT16:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container_using_cast<uint16_t>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_BOOL:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container_using_cast<char>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_INT64:
                tensor_size = tensor_info->int64_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container<int64_t>(tensor_info->int64_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_UINT32:
                tensor_size = tensor_info->uint64_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container_using_cast<uint32_t>(tensor_info->uint64_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_UINT64:
                tensor_size = tensor_info->uint64_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container<uint64_t>(tensor_info->uint64_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN: {
                tensor_size = tensor_info->int32_data_size();
                auto data = std::make_shared<std::vector<ov::float8_e4m3>>();
                data->reserve(tensor_size);
                std::transform(tensor_info->int32_data().begin(),
                               tensor_info->int32_data().end(),
                               std::back_inserter(*data),
                               [](int32_t elem) {
                                   return ov::float8_e4m3::from_bits(static_cast<uint8_t>(elem));
                               });
                tensor_meta_info.m_buffer =
                    std::make_shared<ov::SharedBuffer<std::shared_ptr<std::vector<ov::float8_e4m3>>>>(
                        reinterpret_cast<char*>(data->data()),
                        data->size() * sizeof(ov::float8_e4m3),
                        data);
                break;
            }
            case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2: {
                tensor_size = tensor_info->int32_data_size();
                auto data = std::make_shared<std::vector<ov::float8_e5m2>>();
                data->reserve(tensor_size);
                std::transform(tensor_info->int32_data().begin(),
                               tensor_info->int32_data().end(),
                               std::back_inserter(*data),
                               [](int32_t elem) {
                                   return ov::float8_e5m2::from_bits(static_cast<uint8_t>(elem));
                               });
                tensor_meta_info.m_buffer =
                    std::make_shared<ov::SharedBuffer<std::shared_ptr<std::vector<ov::float8_e5m2>>>>(
                        reinterpret_cast<char*>(data->data()),
                        data->size() * sizeof(ov::float8_e5m2),
                        data);
                break;
            }
            case TensorProto_DataType::TensorProto_DataType_FLOAT16: {
                tensor_size = tensor_info->int32_data_size();
                auto data = std::make_shared<std::vector<ov::float16>>();
                data->reserve(tensor_size);
                std::transform(tensor_info->int32_data().begin(),
                               tensor_info->int32_data().end(),
                               std::back_inserter(*data),
                               [](int32_t elem) {
                                   return ov::float16::from_bits(static_cast<uint16_t>(elem));
                               });
                tensor_meta_info.m_buffer =
                    std::make_shared<ov::SharedBuffer<std::shared_ptr<std::vector<ov::float16>>>>(
                        reinterpret_cast<char*>(data->data()),
                        data->size() * sizeof(ov::float16),
                        data);
                break;
            }
            case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
                tensor_size = tensor_info->int32_data_size();
                tensor_meta_info.m_buffer =
                    make_buffer_from_container_using_cast<ov::bfloat16>(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_FLOAT:
                tensor_size = tensor_info->float_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container<float>(tensor_info->float_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_DOUBLE:
                tensor_size = tensor_info->double_data_size();
                tensor_meta_info.m_buffer = make_buffer_from_container<double>(tensor_info->double_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_STRING: {
                tensor_size = tensor_info->string_data_size();
                auto data = std::make_shared<std::vector<std::string>>(tensor_info->string_data().begin(),
                                                                       tensor_info->string_data().end());
                tensor_meta_info.m_buffer =
                    std::make_shared<ov::SharedBuffer<std::shared_ptr<std::vector<std::string>>>>(
                        reinterpret_cast<char*>(data->data()),
                        data->size() * sizeof(std::string),
                        data);
                break;
            }
            default:
                throw std::runtime_error("Unsupported type " +
                                         ::ONNX_NAMESPACE::TensorProto_DataType_Name(tensor_info->data_type()));
                break;
            }
        }
    }
    if (tensor_meta_info.m_tensor_name == nullptr) {
        tensor_meta_info.m_tensor_name = &empty_name;
    }
    if (tensor_meta_info.m_partial_shape == ov::Shape{0} && tensor_size == 1) {
        tensor_meta_info.m_partial_shape = ov::Shape{};
    }
    return tensor_meta_info;
}

GraphIteratorProto::GraphIteratorProto(const GraphIteratorProtoMemoryManagementMode mode)
    : m_graph(nullptr),
      m_parent(nullptr),
      m_model_dir(nullptr),
      m_mode(mode),
      m_mmap_cache{mode == External_MMAP ? std::make_shared<std::map<std::string, std::shared_ptr<ov::MappedMemory>>>()
                                         : nullptr},
      m_stream_cache{mode == External_Stream ? std::make_shared<std::map<std::string, std::shared_ptr<std::ifstream>>>()
                                             : nullptr},
      m_data_holder{mode == External_Stream ? std::make_shared<std::vector<std::shared_ptr<uint8_t>>>() : nullptr} {}

GraphIteratorProto::GraphIteratorProto(GraphIteratorProto* parent, const GraphProto* graph_def) {
    m_graph = graph_def;
    m_parent = parent;
    m_model_dir = parent->m_model_dir;
    m_mode = parent->m_mode;
    m_mmap_cache = parent->m_mmap_cache;
    m_stream_cache = parent->m_stream_cache;
    m_data_holder = parent->m_data_holder;
    m_model = parent->m_model;
}

void GraphIteratorProto::initialize(const std::string& path) {
    m_model_dir = std::make_shared<std::string>(ov::util::get_directory(path).string());
    try {
        std::ifstream model_file(path, std::ios::binary | std::ios::in);
        FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Could not open the file: \"", path, "\"");

        m_model = std::make_shared<ModelProto>();
        FRONT_END_GENERAL_CHECK(m_model->ParseFromIstream(&model_file), "Model can't be parsed");
        model_file.close();
        if (m_model->has_graph()) {
            m_graph = &m_model->graph();
        } else {
            m_graph = nullptr;
            return;
        }
    } catch (...) {
        m_model.reset();
        m_graph = nullptr;
        node_index = 0;
        m_decoders.clear();
        m_tensors.clear();
        throw;
    }
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void GraphIteratorProto::initialize(const std::wstring& path) {
    initialize(ov::util::wstring_to_string(path));
}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

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
    // In case we have any stored external data - free it before beginning
    if (m_data_holder != nullptr) {
        m_data_holder->clear();
    }
    if (m_stream_cache != nullptr) {
        m_stream_cache->clear();
    }
    node_index = 0;
    if (m_decoders.size() > 0 || m_model == nullptr || m_graph == nullptr)
        return;
    m_decoders.reserve(m_graph->initializer_size() + m_graph->input_size() + m_graph->output_size() +
                       m_graph->node_size());
    for (const auto& value : m_graph->input()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, this, 0, -1);
        m_decoders.push_back(tensor);
        const auto& t_name = *tensor->get_tensor_info().m_tensor_name;
        if (m_tensors.count(t_name) > 0) {
            throw std::runtime_error("Tensor already exists \"" + t_name + "\"");
        }
        m_tensors.emplace(t_name, tensor);
    }
    for (const auto& value : m_graph->output()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, this, -1, 0);
        m_decoders.push_back(tensor);
        const auto& t_name = *tensor->get_tensor_info().m_tensor_name;
        if (m_tensors.count(t_name) == 0) {
            // model may have several outputs of the same tensor
            m_tensors.emplace(t_name, tensor);
        }
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
                extract_tensor_meta_info(&initializer, nullptr, this);
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
        const std::string& domain = node.has_domain() ? node.domain() : DEFAULT_DOMAIN;
        int64_t opset = get_opset_version(domain);
        if (opset == -1) {
            // Forcing a first opset instead of failing
            opset = 1;
        }
        auto decoder_node =
            std::make_shared<DecoderProto>(&node, static_cast<uint64_t>(opset), this, input_tensors, output_tensors);
        m_decoders.push_back(decoder_node);
    }
    // If it is a topmost GraphIterator object, and we are not working in Enable MMAP Mode
    // Then we can close all opened streams which are used for accessing to external data files
    if (m_stream_cache != nullptr && m_parent == nullptr) {
        m_stream_cache->clear();
    }
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
        if ((domain == DEFAULT_DOMAIN && opset_import.domain() == "ai.onnx") ||
            (domain == "ai.onnx" && opset_import.domain() == DEFAULT_DOMAIN) || (domain == opset_import.domain())) {
            return opset_import.version();
        }
    }

    return -1;
}

namespace detail {
namespace {
enum Field {
    IR_VERSION = 1,
    PRODUCER_NAME = 2,
    PRODUCER_VERSION = 3,
    DOMAIN_ = 4,  // DOMAIN collides with some existing symbol in MSVC thus - underscore
    MODEL_VERSION = 5,
    DOC_STRING = 6,
    GRAPH = 7,
    OPSET_IMPORT = 8,
    METADATA_PROPS = 14,
    TRAINING_INFO = 20,
    FUNCTIONS = 25
};

enum WireType { VARINT = 0, BITS_64 = 1, LENGTH_DELIMITED = 2, START_GROUP = 3, END_GROUP = 4, BITS_32 = 5 };

// A PB key consists of a field number (defined in onnx.proto) and a type of data that follows this key
using PbKey = std::pair<char, char>;

// This pair represents a key found in the encoded model and optional size of the payload
// that follows the key (in bytes). The payload should be skipped for fast check purposes.
using ONNXField = std::pair<Field, uint32_t>;

bool is_correct_onnx_field(const PbKey& decoded_key) {
    static const std::map<Field, WireType> onnx_fields = {
        {IR_VERSION, VARINT},
        {PRODUCER_NAME, LENGTH_DELIMITED},
        {PRODUCER_VERSION, LENGTH_DELIMITED},
        {DOMAIN_, LENGTH_DELIMITED},
        {MODEL_VERSION, VARINT},
        {DOC_STRING, LENGTH_DELIMITED},
        {GRAPH, LENGTH_DELIMITED},
        {OPSET_IMPORT, LENGTH_DELIMITED},
        {METADATA_PROPS, LENGTH_DELIMITED},
        {TRAINING_INFO, LENGTH_DELIMITED},
        {FUNCTIONS, LENGTH_DELIMITED},
    };

    if (!onnx_fields.count(static_cast<Field>(decoded_key.first))) {
        return false;
    }

    return onnx_fields.at(static_cast<Field>(decoded_key.first)) == static_cast<WireType>(decoded_key.second);
}

/**
 * Only 7 bits in each component of a varint count in this algorithm. The components form
 * a decoded number when they are concatenated bitwise in reverse order. For example:
 * bytes = [b1, b2, b3, b4]
 * varint = b4 ++ b3 ++ b2 ++ b1  <== only 7 bits of each byte should be extracted before concat
 *
 *             b1         b2
 * bytes = [00101100, 00000010]
 *             b2         b1
 * varint = 0000010 ++ 0101100 = 100101100 => decimal: 300
 * Each consecutive varint byte needs to be left-shifted "7 x its position in the vector"
 * and bitwise added to the accumulator afterward.
 */
uint32_t varint_bytes_to_number(const std::vector<uint8_t>& bytes) {
    uint32_t accumulator = 0u;

    for (size_t i = 0; i < bytes.size(); ++i) {
        uint32_t b = bytes[i];
        b <<= 7 * i;
        accumulator |= b;
    }

    return accumulator;
}

uint32_t decode_varint(std::istream& model) {
    std::vector<uint8_t> bytes;
    // max 4 bytes for a single value because this function returns a 32-bit long decoded varint
    const size_t MAX_VARINT_BYTES = 4u;
    // optimization to avoid allocations during push_back calls
    bytes.reserve(MAX_VARINT_BYTES);

    char key_component = 0;
    model.get(key_component);

    // keep reading all bytes which have the MSB on from the stream
    while (key_component & 0x80 && bytes.size() < MAX_VARINT_BYTES) {
        // drop the most significant bit
        const uint8_t component = key_component & ~0x80;
        bytes.push_back(component);
        model.get(key_component);
    }
    // add the last byte - the one with MSB off
    bytes.push_back(key_component);

    return varint_bytes_to_number(bytes);
}

PbKey decode_key(const char key) {
    // 3 least significant bits
    const char wire_type = key & 0b111;
    // remaining bits
    const char field_number = key >> 3;
    return {field_number, wire_type};
}

ONNXField decode_next_field(std::istream& model) {
    char key = 0;
    model.get(key);

    const auto decoded_key = decode_key(key);

    if (!is_correct_onnx_field(decoded_key)) {
        throw std::runtime_error{"Incorrect field detected in the processed model"};
    }

    const auto onnx_field = static_cast<Field>(decoded_key.first);

    switch (decoded_key.second) {
    case VARINT: {
        // the decoded varint is the payload in this case but its value does not matter
        // in the fast check process so it can be discarded
        decode_varint(model);
        return {onnx_field, 0};
    }
    case LENGTH_DELIMITED:
        // the varint following the key determines the payload length
        return {onnx_field, decode_varint(model)};
    case BITS_64:
        return {onnx_field, 8};
    case BITS_32:
        return {onnx_field, 4};
    case START_GROUP:
    case END_GROUP:
        throw std::runtime_error{"StartGroup and EndGroup are not used in ONNX models"};
    default:
        throw std::runtime_error{"Unknown WireType encountered in the model"};
    }
}

inline void skip_payload(std::istream& model, uint32_t payload_size) {
    model.seekg(payload_size, std::ios::cur);
}
}  // namespace
}  // namespace detail

bool is_valid_model(std::istream& model) {
    // the model usually starts with a 0x08 byte indicating the ir_version value
    // so this checker expects at least 3 valid ONNX keys to be found in the validated model
    const size_t EXPECTED_FIELDS_FOUND = 3u;
    std::unordered_set<detail::Field, std::hash<int>> onnx_fields_found = {};
    try {
        while (!model.eof() && onnx_fields_found.size() < EXPECTED_FIELDS_FOUND) {
            const auto field = detail::decode_next_field(model);

            if (onnx_fields_found.count(field.first) > 0) {
                // if the same field is found twice, this is not a valid ONNX model
                return false;
            } else {
                onnx_fields_found.insert(field.first);
                detail::skip_payload(model, field.second);
            }
        }

        return onnx_fields_found.size() == EXPECTED_FIELDS_FOUND;
    } catch (...) {
        return false;
    }
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
