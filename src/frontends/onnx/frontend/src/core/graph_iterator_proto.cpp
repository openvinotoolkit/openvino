// Copyright (C) 2018-2026 Intel Corporation
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "decoder_proto.hpp"
#include "openvino/frontend/graph_iterator.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/wstring_convert_util.hpp"
#include "transform.hpp"

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

const ::ONNX_NAMESPACE::TypeProto_Tensor* get_tensor_type(const ::ONNX_NAMESPACE::TypeProto& type_proto) {
    if (type_proto.has_tensor_type()) {
        return &type_proto.tensor_type();
    }
    if (type_proto.has_optional_type() && type_proto.optional_type().has_elem_type()) {
        return get_tensor_type(type_proto.optional_type().elem_type());
    }
    if (type_proto.has_sequence_type() && type_proto.sequence_type().has_elem_type()) {
        return get_tensor_type(type_proto.sequence_type().elem_type());
    }
    return nullptr;
}

void fixup_legacy_nodes(::ONNX_NAMESPACE::ModelProto& model_proto) {
    auto* graph_proto = model_proto.mutable_graph();
    if (!graph_proto) {
        return;
    }
    constexpr const char legacy_domain[] = "org.openvinotoolkit";

    for (auto& node : *graph_proto->mutable_node()) {
        const auto needs_fix = std::find(ov::frontend::onnx::transform::legacy_ops_to_fixup.begin(),
                                         ov::frontend::onnx::transform::legacy_ops_to_fixup.end(),
                                         node.op_type()) != ov::frontend::onnx::transform::legacy_ops_to_fixup.end();
        if (!needs_fix) {
            continue;
        }

        if (!node.has_domain() || node.domain().empty() || node.domain() == "ai.onnx") {
            node.set_domain(legacy_domain);
        }
    }
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
        ov::util::get_absolute_file_path(ov::util::path_join({graph_iterator->get_model_dir(), ext_location}));
    const int64_t file_size = ov::util::file_size(full_path);
    if ((file_size <= 0 && ext_data_length > 0) ||
        ext_data_offset + ext_data_length > static_cast<uint64_t>(file_size)) {
        // not_existed_file.data, offset: 4096, data_length: 16)
        std::stringstream ss;
        ss << "Invalid usage of method for externally stored data in file (" << ext_location;
        ss << ", offset: " << ext_data_offset << ", data_length: " << ext_data_length << ")";
        throw std::runtime_error(ss.str());
    }
    const size_t resolved_data_length = ext_data_length > 0
                                            ? static_cast<size_t>(ext_data_length)
                                            : static_cast<size_t>(file_size) - static_cast<size_t>(ext_data_offset);
    auto memory_mode = graph_iterator->get_memory_management_mode();
    // Remove when cache map will use path instead string.
    const auto full_path_str = ov::util::path_to_string(full_path);
    if (ext_location == "*/_ORT_MEM_ADDR_/*") {
        // Specific ONNX Runtime Case when it passes a model with self-managed data
        tensor_meta_info.m_is_raw = true;
        tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset);
        tensor_meta_info.m_tensor_data_size = ext_data_length;
        return true;
    } else if (memory_mode == External_MMAP) {
        auto cache = graph_iterator->get_mmap_cache();
        auto cached_mapped_memory = cache->find(full_path_str);
        std::shared_ptr<ov::MappedMemory> mapped_memory;
        if (cached_mapped_memory != cache->end()) {
            mapped_memory = cached_mapped_memory->second;
        } else {
            mapped_memory = ov::load_mmap_object(full_path);
            (*cache)[full_path_str] = mapped_memory;
        }
        tensor_meta_info.m_is_raw = true;
        tensor_meta_info.m_tensor_data =
            static_cast<uint8_t*>(static_cast<void*>(mapped_memory->data() + ext_data_offset));
        tensor_meta_info.m_tensor_data_size = resolved_data_length;
        return true;
    } else if (memory_mode == External_Stream) {
        auto cache = graph_iterator->get_stream_cache();
        FRONT_END_GENERAL_CHECK(cache, "Stream cache is not initialized for external stream mode");
        auto cached_stream = cache->find(full_path_str);
        std::shared_ptr<std::ifstream> external_data_stream;
        if (cached_stream != cache->end()) {
            external_data_stream = cached_stream->second;
        } else {
            external_data_stream = {new std::ifstream(full_path, std::ios::binary | std::ios::in | std::ios::ate),
                                    [](std::ifstream* p) {
                                        p->close();
                                        delete p;
                                    }};
            (*cache)[full_path_str] = external_data_stream;
        }

        if (external_data_stream->fail() || !external_data_stream->good()) {
            throw std::runtime_error("Failed to open external data stream");
        }

        tensor_meta_info.m_is_raw = true;
        tensor_meta_info.m_tensor_data_size = resolved_data_length;
        uint8_t* data_ptr = graph_iterator->allocate_data(tensor_meta_info.m_tensor_data_size).get();
        tensor_meta_info.m_tensor_data = data_ptr;

        // default value of m_offset is 0
        external_data_stream->seekg(ext_data_offset, std::ios::beg);

        external_data_stream->read(static_cast<char*>(static_cast<void*>(data_ptr)),
                                   tensor_meta_info.m_tensor_data_size);
        return true;
    } else if (memory_mode == Internal_MMAP || memory_mode == Internal_Stream) {
        tensor_meta_info.m_external_location = std::make_shared<std::string>(full_path_str);
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
    const auto* graph_def = graph_iterator->get_graph();
    if ((tensor_info == nullptr && value_info == nullptr) || graph_def == nullptr) {
        throw std::runtime_error("Wrong usage");
    }

    ov::frontend::onnx::TensorMetaInfo tensor_meta_info{};
    tensor_meta_info.m_external_location = nullptr;
    tensor_meta_info.m_is_raw = false;

    if (tensor_info != nullptr) {
        tensor_meta_info.m_tensor_name = tensor_info->has_name() ? &tensor_info->name() : &empty_name;
        tensor_meta_info.m_partial_shape =
            ov::PartialShape(std::vector<int64_t>(tensor_info->dims().begin(), tensor_info->dims().end()));
        tensor_meta_info.m_element_type =
            tensor_info->has_data_type() ? get_ov_element_type(tensor_info->data_type()) : ov::element::dynamic;
        if (tensor_info->has_segment()) {
            throw std::runtime_error("Loading segments isn't supported");
        }
        if (tensor_info->has_data_location() &&
            tensor_info->data_location() == TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
            if (extract_tensor_external_data(tensor_meta_info, tensor_info, graph_iterator)) {
                return tensor_meta_info;
            }
            throw std::runtime_error("Unsupported method for externally stored data");
        }
        if (tensor_info->has_raw_data()) {
            tensor_meta_info.m_tensor_data = reinterpret_cast<const uint8_t*>(tensor_info->raw_data().data());
            tensor_meta_info.m_tensor_data_size = tensor_info->raw_data().size();
            tensor_meta_info.m_is_raw = true;
        } else {
            const auto assign_numeric_data = [&](const auto& container) {
                tensor_meta_info.m_tensor_data = reinterpret_cast<const uint8_t*>(container.data());
                tensor_meta_info.m_tensor_data_size = static_cast<size_t>(container.size());
            };
            switch (tensor_info->data_type()) {
            case TensorProto_DataType::TensorProto_DataType_FLOAT:
                assign_numeric_data(tensor_info->float_data());
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
                assign_numeric_data(tensor_info->int32_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_INT64:
                assign_numeric_data(tensor_info->int64_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_UINT32:
            case TensorProto_DataType::TensorProto_DataType_UINT64:
                assign_numeric_data(tensor_info->uint64_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_DOUBLE:
                assign_numeric_data(tensor_info->double_data());
                break;
            case TensorProto_DataType::TensorProto_DataType_STRING:
                tensor_meta_info.m_tensor_data_any =
                    std::vector<std::string>(tensor_info->string_data().begin(), tensor_info->string_data().end());
                tensor_meta_info.m_tensor_data_size = tensor_info->string_data_size();
                break;
            default:
                throw std::runtime_error("Unsupported type " +
                                         ::ONNX_NAMESPACE::TensorProto_DataType_Name(tensor_info->data_type()));
            }
        }
    } else if (value_info != nullptr) {
        tensor_meta_info.m_tensor_name = value_info->has_name() ? &value_info->name() : &empty_name;
        const auto* value_type = value_info->has_type() ? get_tensor_type(value_info->type()) : nullptr;
        const auto value_case =
            value_info->has_type() ? value_info->type().value_case() : ::ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
        if (value_type == nullptr && value_case != ::ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET) {
            throw std::runtime_error("Unsupported value_info type: " + (*tensor_meta_info.m_tensor_name));
        }
        if (value_type != nullptr && value_type->has_shape()) {
            std::vector<int64_t> dims;
            const auto& shape = value_type->shape();
            dims.reserve(shape.dim_size());
            for (const auto& dim : shape.dim()) {
                dims.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
            }
            tensor_meta_info.m_partial_shape = ov::PartialShape{dims};
        } else {
            tensor_meta_info.m_partial_shape = ov::PartialShape::dynamic();
        }
        tensor_meta_info.m_element_type = (value_type != nullptr && value_type->has_elem_type())
                                              ? get_ov_element_type(value_type->elem_type())
                                              : ov::element::dynamic;
    }

    if (tensor_meta_info.m_tensor_name == nullptr) {
        tensor_meta_info.m_tensor_name = &empty_name;
    }
    if (tensor_meta_info.m_partial_shape == ov::Shape{0} && tensor_meta_info.m_tensor_data_size == 1) {
        tensor_meta_info.m_partial_shape = ov::Shape{};
    }
    return tensor_meta_info;
}

GraphIteratorProto::GraphIteratorProto(const GraphIteratorProtoMemoryManagementMode mode)
    : m_graph(nullptr),
      m_parent(nullptr),
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

void GraphIteratorProto::initialize(const std::filesystem::path& path) {
    m_model_dir = ov::util::get_directory(path);
    const auto path_string = ov::util::path_to_string(path);
    try {
        std::ifstream model_file(path, std::ios::binary | std::ios::in);
        FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Could not open the file: \"", path_string, "\"");

        m_model = std::make_shared<ModelProto>();
        FRONT_END_GENERAL_CHECK(m_model->ParseFromIstream(&model_file), "Model can't be parsed");
        model_file.close();
        if (m_model->has_graph()) {
            fixup_legacy_nodes(*m_model);
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
std::shared_ptr<DecoderProtoTensor> GraphIteratorProto::get_tensor(const std::string& name,
                                                                   GraphIteratorProto** owner) {
    if (m_tensors.count(name) == 0) {
        if (name == empty_name) {
            *owner = this;
            const auto& tensor_decoder = std::make_shared<DecoderProtoTensor>(empty_name, this);
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
    const auto& graph = *m_graph;
    m_decoders.reserve(graph.initializer_size() + graph.input_size() + graph.output_size() + graph.node_size());

    std::unordered_map<std::string, const TensorProto*> initializer_by_name;
    initializer_by_name.reserve(graph.initializer_size());
    for (const auto& initializer : graph.initializer()) {
        if (initializer.has_name()) {
            initializer_by_name.emplace(initializer.name(), &initializer);
        }
    }

    std::unordered_map<std::string, const ValueInfoProto*> value_info_by_name;
    value_info_by_name.reserve(graph.value_info().size());
    for (const auto& info : graph.value_info()) {
        if (info.has_name()) {
            value_info_by_name.emplace(info.name(), &info);
        }
    }

    int64_t index = 0;
    for (const auto& value : graph.input()) {
        std::shared_ptr<DecoderProtoTensor> tensor;
        if (auto init_it = initializer_by_name.find(value.name()); init_it == initializer_by_name.end()) {
            tensor = std::make_shared<DecoderProtoTensor>(&value, this, index++, -1);
        } else {
            tensor = std::make_shared<DecoderProtoTensor>(init_it->second, this);
        }
        const auto& tensor_name = *tensor->get_tensor_info().m_tensor_name;
        if (auto [it, inserted] = m_tensors.try_emplace(tensor_name, tensor); !inserted) {
            throw std::runtime_error("Tensor already exists \"" + tensor_name + "\"");
        }
        m_decoders.push_back(tensor);
    }

    index = 0;
    for (const auto& value : graph.output()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, this, -1, index++);
        const auto& tensor_name = *tensor->get_tensor_info().m_tensor_name;
        m_tensors.try_emplace(tensor_name, tensor);
        m_decoders.push_back(tensor);
    }

    for (const auto& initializer : graph.initializer()) {
        if (auto tensor_it = m_tensors.find(initializer.name()); tensor_it != m_tensors.end()) {
            *const_cast<ov::frontend::onnx::TensorMetaInfo*>(&tensor_it->second->get_tensor_info()) =
                extract_tensor_meta_info(&initializer, nullptr, this);
        } else {
            m_tensors.try_emplace(initializer.name(), std::make_shared<DecoderProtoTensor>(&initializer, this, -1, -1));
        }
    }
    size_t top_index = 0;
    for (const auto& node : graph.node()) {
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
            if (name.empty()) {
                output_tensors.push_back(&this->get_tensor(empty_name, &tensor_owner)->get_tensor_info());
                continue;
            }
            if (auto tensor_it = m_tensors.find(name); tensor_it == m_tensors.end()) {
                std::shared_ptr<DecoderProtoTensor> tensor;
                if (auto init_it = initializer_by_name.find(name); init_it != initializer_by_name.end()) {
                    tensor = std::make_shared<DecoderProtoTensor>(init_it->second, this, -1, -1);
                } else if (auto value_info_it = value_info_by_name.find(name);
                           value_info_it != value_info_by_name.end()) {
                    tensor = std::make_shared<DecoderProtoTensor>(value_info_it->second, this, -1, -1);
                } else {
                    tensor = std::make_shared<DecoderProtoTensor>(name, this, -1, -1);
                }
                m_decoders.push_back(tensor);
                const auto& inserted_tensor = m_tensors.try_emplace(name, tensor).first->second;
                output_tensors.push_back(&inserted_tensor->get_tensor_info());
            } else {
                output_tensors.push_back(&tensor_it->second->get_tensor_info());
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

std::map<std::string, std::string> GraphIteratorProto::get_metadata() const {
    std::map<std::string, std::string> metadata;

    if (!m_model) {
        return metadata;
    }

    const auto& model_metadata = m_model->metadata_props();
    for (const auto& prop : model_metadata) {
        metadata.emplace(prop.key(), prop.value());
    }
    return metadata;
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
