// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_proto.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <openvino/frontend/graph_iterator.hpp>

#include "decoder_proto.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/file_util.hpp"
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

namespace {
bool extract_tensor_external_data(ov::frontend::onnx::TensorMetaInfo& tensor_meta_info,
                                  const TensorProto* tensor_info,
                                  GraphIteratorProto* graph_iterator) {
    std::string ext_location{};
    uint64_t ext_data_offset = 0;
    uint64_t ext_data_length = 0;
    std::string m_sha1_digest{};  // for future use
    for (const auto& entry : tensor_info->external_data()) {
        if (entry.key() == "location") {
            ext_location = ov::util::sanitize_path(entry.value());
        } else if (entry.key() == "offset") {
            ext_data_offset = std::stoull(entry.value());
        } else if (entry.key() == "length") {
            ext_data_length = std::stoull(entry.value());
        } else if (entry.key() == "checksum") {
            m_sha1_digest = entry.value();
        }
    }
    const auto full_path =
        ov::util::get_absolute_file_path(ov::util::path_join({graph_iterator->get_model_dir(), ext_location}).string());
    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size <= 0 || ext_data_offset + ext_data_length > static_cast<uint64_t>(file_size)) {
        // not_existed_file.data, offset: 4096, data_length: 16)
        std::stringstream ss;
        ss << "Invalid usage of method for externally stored data in file (" << ext_location;
        ss << ", offset: " << ext_data_offset << ", data_length: " << ext_data_length << ")";
        throw std::runtime_error(ss.str());
    }
    auto memory_mode = graph_iterator->get_memory_management_mode();
    if (ext_location == "*/_ORT_MEM_ADDR_/*") {
        // Specific ONNX Runtime Case when it passes a model with self-managed data
        tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset);
        tensor_meta_info.m_tensor_data_size = ext_data_length;
        return true;
    } else if (memory_mode == External_MMAP) {
        auto cache = graph_iterator->get_mmap_cache();
        auto cached_mapped_memory = cache->find(full_path);
        std::shared_ptr<ov::MappedMemory> mapped_memory;
        if (cached_mapped_memory != cache->end()) {
            mapped_memory = cached_mapped_memory->second;
        } else {
            mapped_memory = ov::load_mmap_object(full_path);
            (*cache)[full_path] = mapped_memory;
        }
        tensor_meta_info.m_tensor_data =
            static_cast<uint8_t*>(static_cast<void*>(mapped_memory->data() + ext_data_offset));
        tensor_meta_info.m_tensor_data_size =
            ext_data_length > 0 ? ext_data_length : static_cast<size_t>(file_size) - ext_data_length;
        return true;
    } else if (memory_mode == External_Stream) {
        auto cache = graph_iterator->get_stream_cache();
        auto cached_stream = cache->find(full_path);
        std::shared_ptr<std::ifstream> external_data_stream;
        if (cached_stream != cache->end()) {
            external_data_stream = cached_stream->second;
        } else {
            external_data_stream = {
                new std::ifstream(full_path.c_str(), std::ios::binary | std::ios::in | std::ios::ate),
                [](std::ifstream* p) {
                    p->close();
                    delete p;
                }};
            (*cache)[full_path] = external_data_stream;
        }

        if (external_data_stream->fail() || !external_data_stream->good()) {
            throw std::runtime_error("Failed to open external data stream");
        }

        tensor_meta_info.m_tensor_data_size =
            ext_data_length > 0 ? ext_data_length : static_cast<size_t>(file_size) - ext_data_length;
        uint8_t* data_ptr = graph_iterator->allocate_data(tensor_meta_info.m_tensor_data_size).get();
        tensor_meta_info.m_tensor_data = data_ptr;

        // default value of m_offset is 0
        external_data_stream->seekg(ext_data_offset, std::ios::beg);

        external_data_stream->read(static_cast<char*>(static_cast<void*>(data_ptr)),
                                   tensor_meta_info.m_tensor_data_size);
        return true;
    } else if (memory_mode == Internal_MMAP || memory_mode == Internal_Stream) {
        tensor_meta_info.m_external_location = std::make_shared<std::string>(full_path);
        tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset);
        tensor_meta_info.m_tensor_data_size = ext_data_length;
        return true;
    } else {
        throw std::runtime_error("Unsupported memory management mode");
    }
}
}  // namespace

ov::frontend::onnx::TensorMetaInfo extract_tensor_meta_info(const TensorProto* tensor_info,
                                                            const ValueInfoProto* value_info,
                                                            GraphIteratorProto* graph_iterator) {
    auto graph_def = graph_iterator->get_graph();
    ov::frontend::onnx::TensorMetaInfo tensor_meta_info{};
    tensor_meta_info.m_external_location = nullptr;
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
        std::vector<int64_t> dims(tensor_info->dims().begin(), tensor_info->dims().end());
        if (dims.size() == 0 || (dims.size() == 1 && dims[0] == 0)) {
            tensor_meta_info.m_partial_shape = ov::PartialShape{};
        } else {
            tensor_meta_info.m_partial_shape = ov::PartialShape{dims};
        }
        tensor_meta_info.m_element_type =
            tensor_info->has_data_type() ? get_ov_element_type(tensor_info->data_type()) : ov::element::dynamic;
        if (tensor_info->has_data_location() &&
            tensor_info->data_location() == TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
            if (extract_tensor_external_data(tensor_meta_info, tensor_info, graph_iterator)) {
                return tensor_meta_info;
            }
            throw std::runtime_error("Unsupported method for externally stored data");
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
        case TensorProto_DataType::TensorProto_DataType_UINT32:
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
    if (tensor_meta_info.m_tensor_name == nullptr) {
        tensor_meta_info.m_tensor_name = &empty_name;
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
    } catch (...) {
        m_model.reset();
        m_graph = nullptr;
        node_index = 0;
        m_decoders.clear();
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
        const std::string& domain = node.has_domain() && node.domain() != "ai.onnx" ? node.domain() : DEFAULT_DOMAIN;
        int64_t opset = get_opset_version(domain);
        if (opset == -1) {
            // Forcing a first opset instead of failing
            opset = 1;
            // throw std::runtime_error("Operation version isn't found");
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
        if (domain == opset_import.domain()) {
            return opset_import.version();
        }
    }

    return -1;
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <unordered_set>
#include <vector>

namespace ov {
namespace frontend {
namespace onnx {
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