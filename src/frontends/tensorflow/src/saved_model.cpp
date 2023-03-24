// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <fstream>
#include <string>

#include "graph_iterator_saved_model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "tensor_bundle.pb.h"
#include "trackable_object_graph.pb.h"

#ifdef ENABLE_SNAPPY_COMPRESSION
#    include "snappy.h"
#endif

namespace ov {
namespace frontend {
namespace tensorflow {

template <typename T>
static T smReadFixed(const char* ptr) {
    T result = 0;
    for (uint8_t i = 0; i < sizeof(T); ++i) {
        result |= ptr[i] << (i * 8);
    }
    return result;
}

template <typename T>
static T smUnpack(char*& ptr, const char* ptr_end) {
    T result = 0;
    for (uint8_t i = 0; i < sizeof(T) * 7 && ptr < ptr_end; i += 7) {
        T byte = *(ptr++);
        if (byte & 0x80) {
            result |= ((byte & 0x7F) << i);
        } else {
            result |= byte << i;
            return result;
        }
    }
    return 0;
}

struct VIBlock {
    uint64_t m_size;
    uint64_t m_offset;

    void read(char*& ptr, const char* ptr_end) {
        m_offset = smUnpack<uint64_t>(ptr, ptr_end);
        m_size = smUnpack<uint64_t>(ptr, ptr_end);
    }
};

struct VIFooter {
    VIBlock m_metaIndex;
    VIBlock m_index;

    void read(char*& ptr, const char* ptr_end) {
        m_index.read(ptr, ptr_end);
        m_metaIndex.read(ptr, ptr_end);
    }

    void read(std::ifstream& fs) {
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg();
        char footerData[48] = {}, *ptr = &footerData[0];
        fs.seekg(size - sizeof(footerData));
        fs.read(ptr, sizeof(footerData));

        // https://github.com/tensorflow/tensorflow/blob/9659b7bdca80a8ef8240eb021d4da089034eeb00/tensorflow/tsl/lib/io/format.cc#L59
        ptr += sizeof(footerData) - 8;
        uint32_t magic_lo = *reinterpret_cast<const uint32_t*>(ptr);
        uint32_t magic_hi = *reinterpret_cast<const uint32_t*>(ptr + 4);
        uint64_t magic_no = (static_cast<uint64_t>(magic_hi) << 32) | static_cast<uint64_t>(magic_lo);

        FRONT_END_GENERAL_CHECK(magic_no == 0xdb4775248b80fb57ull, "Wrong index file, magic number mismatch detected");

        ptr = &footerData[0];
        m_metaIndex.read(ptr, ptr + sizeof(footerData));
        m_index.read(ptr, ptr + sizeof(footerData));
    }
};

void SavedModelVariablesIndex::read_variables_index_block(std::ifstream& fs,
                                                          const VIBlock& index,
                                                          std::vector<char>& data,
                                                          uint32_t& offset,
                                                          uint32_t& offset_end) {
    size_t block_size = index.m_size;
    data.clear();
    data.resize(block_size + 5 /*kBlockTrailerSize*/);
    fs.seekg(index.m_offset, std::ios::beg);
    fs.read(data.data(), data.size());
#ifndef ENABLE_SNAPPY_COMPRESSION
    FRONT_END_GENERAL_CHECK(data[block_size] == 0, "Compressed files aren't supported");
#else
    FRONT_END_GENERAL_CHECK(data[block_size] == 0 || data[block_size] == 1, "Compression method isn't supported");
    if (data[block_size] == 1) {
        size_t uncompressed_length = 0;
        FRONT_END_GENERAL_CHECK(snappy::GetUncompressedLength(data.data(), data.size(), &uncompressed_length),
                                "Cannot retrieve uncompressed block length");
        std::string uncompressed_string;
        uncompressed_string.reserve(uncompressed_length);
        snappy::Uncompress(data.data(), data.size(), &uncompressed_string);
        data.resize(uncompressed_length);
        std::copy(uncompressed_string.begin(), uncompressed_string.end(), data.begin());
        block_size = uncompressed_length;
    }
#endif
    uint32_t numRestarts = smReadFixed<uint32_t>(data.data() + block_size - sizeof(uint32_t));
    size_t maxRestarts = (block_size - sizeof(uint32_t)) / sizeof(uint32_t);
    FRONT_END_GENERAL_CHECK(maxRestarts >= numRestarts, "Wrong restarts value");
    offset_end = static_cast<uint32_t>(block_size) - ((numRestarts + 1) * sizeof(uint32_t));
    offset = smReadFixed<uint32_t>(data.data() + offset_end);
}

void SavedModelVariablesIndex::read_variables_index_pair(char*& ptr,
                                                         const char* ptr_end,
                                                         std::string& key,
                                                         char*& value,
                                                         uint32_t& val_length) {
    uint32_t shared, nonShared;
    shared = smUnpack<uint32_t>(ptr, ptr_end);
    nonShared = smUnpack<uint32_t>(ptr, ptr_end);
    val_length = smUnpack<uint32_t>(ptr, ptr_end);

    // Key inherits last part of string (shared-size bytes) and appends new string
    // shared_part_key1 //resize(0) + append(shared_part_key1)
    // ............key2 //resize(12) + append(key2)
    // ............key3 //resize(12) + append(key3)
    // new_shared_key4 //resize(0) + append(new_shared_key4)
    // ...........key5 //resize(11) + append(key5)
    key.resize(shared);
    key.append(ptr, nonShared);

    value = ptr + nonShared;
    ptr = value + val_length;
}

void SavedModelVariablesIndex::read_variables_index(std::ifstream& fs,
                                                    std::map<std::string, std::vector<char>>& varIndex) {
    VIFooter footer;

    footer.read(fs);

    std::vector<VIBlock> secondLevel;
    std::vector<char> blockData;

    uint32_t offset = 0, offset_end = 0;

    read_variables_index_block(fs, footer.m_index, blockData, offset, offset_end);
    char *ptr = blockData.data() + offset, *ptr_end = blockData.data() + offset_end, *value = nullptr;
    std::string key = "";
    uint32_t valLength;

    while (ptr < ptr_end) {
        read_variables_index_pair(ptr, ptr_end, key, value, valLength);

        VIBlock valBlock;
        valBlock.read(value, value + valLength);
        secondLevel.push_back(valBlock);
        ptr = value + valLength;
    }

    for (auto& block : secondLevel) {
        read_variables_index_block(fs, block, blockData, offset, offset_end);

        key = "";
        ptr = blockData.data() + offset;
        ptr_end = blockData.data() + offset_end;
        while (ptr < ptr_end) {
            read_variables_index_pair(ptr, ptr_end, key, value, valLength);
            varIndex[key] = std::vector<char>(value, value + valLength);
        }
    }
}

void SavedModelVariablesIndex::read_bundle_header() {
    auto item = m_variables_index.find("");
    FRONT_END_GENERAL_CHECK(item != m_variables_index.end(), "Bundle Header isn't found in index");

    ::tensorflow::BundleHeaderProto bundleHeader;
    FRONT_END_GENERAL_CHECK(bundleHeader.ParseFromString(item->second.data()),
                            "Bundle Header: Cannot parse Bundle Header");
    FRONT_END_GENERAL_CHECK(bundleHeader.version().producer() == 1, "Bundle Header: Unsupported producer version");
    FRONT_END_GENERAL_CHECK(bundleHeader.version().min_consumer() == 0, "Bundle Header: Unsupported consumer version");
    FRONT_END_GENERAL_CHECK(bundleHeader.endianness() == 0, "Bundle Header: BIG endian isn't supported");

    m_total_shards = bundleHeader.num_shards();
}

void SavedModelVariablesIndex::read_checkpointable_object_graph() {
    m_variables_map.clear();

    auto item = m_variables_index.find("_CHECKPOINTABLE_OBJECT_GRAPH");
    FRONT_END_GENERAL_CHECK(item != m_variables_index.end(), "Checkpointable Object Graph isn't found in index");

    ::tensorflow::BundleEntryProto entry;
    FRONT_END_GENERAL_CHECK(entry.ParseFromArray(item->second.data(), static_cast<int>(item->second.size())),
                            "CMO: Cannot parse Bundle Entry");

    FRONT_END_GENERAL_CHECK(entry.slices().empty(), "CMO: Slices are not supported");

    auto shard = m_data_files.find(entry.shard_id());
    FRONT_END_GENERAL_CHECK(shard != m_data_files.end(), "CMO: data files isn't found");

    std::vector<char> data(entry.size());
    ::tensorflow::TrackableObjectGraph tog;

    // TODO: have to understand this offset
    // It looks like reinterpret_cast artifact
    // https://github.com/tensorflow/tensorflow/blob/d90f1947ebcf510b23c238f43c2191e5b3817cb3/tensorflow/cc/experimental/libexport/load.cc#L70
    int chg = 6;
    shard->second->seekg(entry.offset() + chg);
    shard->second->read(data.data(), entry.size() - chg);

    // Might be need to remove this verification:
    // https://github.com/tensorflow/tensorflow/blob/d90f1947ebcf510b23c238f43c2191e5b3817cb3/tensorflow/cc/experimental/libexport/load.cc#L73
    // FRONT_END_GENERAL_CHECK(tog.ParseFromArray(data.data(), static_cast<int>(data.size()) - chg), "CMO: Trackable
    // Object Graph couldn't be read");

    tog.ParseFromArray(data.data(), static_cast<int>(data.size()) - chg);

    for (const auto& node : tog.nodes()) {
        for (const auto& attr : node.attributes()) {
            m_variables_map[attr.full_name()] = attr.checkpoint_key();
        }
    }
}

bool GraphIteratorSavedModel::is_valid_signature(const ::tensorflow::SignatureDef& signature) const {
    const std::map<::tensorflow::DataType, ov::element::Type> types{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16},
        {::tensorflow::DataType::DT_STRING, ov::element::undefined}};

    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    return true;
}

bool SavedModelVariablesIndex::read_variables(std::ifstream& vi_stream, const std::string& path) {
    m_variables_index.clear();
    read_variables_index(vi_stream, m_variables_index);
    read_bundle_header();

    std::vector<char> suffix(20);
    for (int32_t shard = 0; shard < m_total_shards; ++shard) {
        std::snprintf(suffix.data(), suffix.size(), "data-%05d-of-%05d", shard, m_total_shards);
        std::string fullPath = ov::util::path_join({path, "variables", std::string("variables.") + suffix.data()});
        m_data_files[shard] =
            std::shared_ptr<std::ifstream>(new std::ifstream(fullPath, std::ifstream::in | std::ifstream::binary));
        FRONT_END_GENERAL_CHECK(m_data_files[shard]->is_open(), "Saved Model's variable index file does not exist");
    }

    read_checkpointable_object_graph();
    return true;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool SavedModelVariablesIndex::read_variables(std::ifstream& vi_stream, const std::wstring& path) {
    m_variables_index.clear();
    read_variables_index(vi_stream, m_variables_index);
    read_bundle_header();

    std::vector<wchar_t> suffix(20);
    for (int32_t shard = 0; shard < m_total_shards; ++shard) {
        swprintf_s(suffix.data(), suffix.size(), L"data-%05d-of-%05d", shard, m_total_shards);
        std::wstring fullPath =
            ov::util::path_join_w({path, L"variables", std::wstring(L"variables.") + suffix.data()});
        m_data_files[shard] =
            std::shared_ptr<std::ifstream>(new std::ifstream(fullPath, std::ifstream::in | std::ifstream::binary));
        FRONT_END_GENERAL_CHECK(m_data_files[shard]->is_open(), "Saved Model's variable index file does not exist");
    }

    read_checkpointable_object_graph();
    return true;
}
#endif

struct PtrNode {
    const ::tensorflow::NodeDef* node;
    std::vector<PtrNode*> inputs;
    std::vector<PtrNode*> outputs;

    PtrNode() : node(nullptr), inputs(), outputs() {}

    PtrNode(const ::tensorflow::NodeDef& src_node, const std::map<std::string, PtrNode*>& node_dictionary) {
        node = &src_node;
        std::vector<std::string> parsedName;
        for (const auto& input_name : node->input()) {
            parse_node_name(input_name, parsedName);

            auto input_node = node_dictionary.find(parsedName[0]);
            if (input_node == node_dictionary.end()) {
                continue;
            }

            input_node->second->outputs.push_back(this);
            inputs.push_back(input_node->second);
        }
    }

    void find_parent_by_op(const std::string& op, std::vector<PtrNode*>& result) const {
        for (auto input : inputs) {
            if (input->op() == op) {
                result.push_back(input);
            }
            input->find_parent_by_op(op, result);
        }
    }

    static void parse_node_name(const std::string& name, std::vector<std::string>& result) {
        result.clear();
        size_t left_pos = name.find_first_of('^'), right_pos = name.find(':');
        if (left_pos != std::string::npos && left_pos < right_pos) {
            ++left_pos;
        } else {
            left_pos = 0;
        }
        while (right_pos != std::string::npos && right_pos > left_pos) {
            result.push_back(name.substr(left_pos, right_pos - left_pos));
            left_pos = right_pos + 1;
            right_pos = name.find(':', left_pos);
        }
        result.push_back(name.substr(left_pos, name.length() - left_pos));
    }

    const std::string& op() const {
        return node->op();
    }
};

static void read_stateful_partitioned_call(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                                           const ::tensorflow::NodeDef& partCall,
                                           std::map<std::string, PtrNode*>& node_dictionary) {
    FRONT_END_GENERAL_CHECK(partCall.op() == "StatefulPartitionedCall", "Passed node isn't StatefulPartitionedCall");

    std::string func_name = partCall.attr().at("f").func().name();

    const ::tensorflow::FunctionDef* func_def = nullptr;
    for (const auto& func : graph_def->library().function()) {
        if (func.signature().name() == func_name) {
            func_def = &func;
            break;
        }
    }

    FRONT_END_GENERAL_CHECK(func_def, "Function isn't found in the library");
    FRONT_END_GENERAL_CHECK(graph_def->has_library(), "GraphDef contains functions, but doesn't have the library");

    std::map<std::string, PtrNode*> nodes;

    // Filling temporary input nodes for exact function
    for (int i = 0; i < func_def->signature().input_arg_size(); ++i) {
        const auto& input_arg = func_def->signature().input_arg(i).name();
        const auto& parent_input = partCall.input(i);
        auto input_node = node_dictionary.find(parent_input);
        if (input_node != node_dictionary.end()) {
            nodes[input_arg] = input_node->second;
        }
    }

    // Parsing nodes and inline partitioned calls
    for (const auto& node : func_def->node_def()) {
        nodes[node.name()] = new PtrNode(node, nodes);

        if (node.op() == "StatefulPartitionedCall") {
            read_stateful_partitioned_call(graph_def, node, nodes);
        }
    }

    // Removing temporary input nodes
    for (int i = 0; i < func_def->signature().input_arg_size(); ++i) {
        const auto& input_arg = func_def->signature().input_arg(i).name();
        auto input_node = nodes.find(input_arg);
        if (input_node != nodes.end()) {
            nodes.erase(input_node);
        }
    }

    // Moving nodes to the global dictionary
    for (const auto& node : nodes) {
        std::string global_name = partCall.name() + "/" + node.first;
        node_dictionary[global_name] = node.second;
    }
}

void GraphIteratorSavedModel::map_assignvariable(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                                                 std::map<std::string, std::string>& variables_map) const {
    std::map<std::string, PtrNode*> nodes;

    for (const auto& node : graph_def->node()) {
        nodes[node.name()] = new PtrNode(node, nodes);

        if (node.op() == "StatefulPartitionedCall") {
            read_stateful_partitioned_call(graph_def, node, nodes);
        }
    }

    for (const auto& node : nodes) {
        if (node.second->op() != "AssignVariableOp") {
            continue;
        }

        // TODO: assets reading

        std::vector<PtrNode*> restorev2_nodes;
        std::vector<PtrNode*> varhandle_nodes;

        node.second->find_parent_by_op("RestoreV2", restorev2_nodes);
        node.second->find_parent_by_op("VarHandleOp", varhandle_nodes);

        FRONT_END_GENERAL_CHECK(restorev2_nodes.size() == 1, "Found unexpected amount of RestoreV2 nodes");
        FRONT_END_GENERAL_CHECK(varhandle_nodes.size() == 1, "Found unexpected amount of VarHandleOp nodes");

        std::vector<std::string> restore_output;
        // Expected path is: RestoreV2 -(output_index)-(0)-> Identity -(0)-(1)-> AssignVariableOp
        PtrNode::parse_node_name(node.second->inputs[1]->node->input(0), restore_output);

        int output_index = std::atoi(restore_output[restore_output.size() - 1].c_str());

        // Expected path is: Const(tensor_names) -(0)-(1)-> RestoreV2
        const auto& variable_name =
            restorev2_nodes[0]->inputs[1]->node->attr().at("value").tensor().string_val(output_index);

        variables_map[varhandle_nodes[0]->node->name()] = variable_name;
    }

    nodes.clear();
}

bool GraphIteratorSavedModel::is_supported(const std::string& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join({path, "saved_model.pb"}));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool GraphIteratorSavedModel::is_supported(const std::wstring& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join_w({path, L"saved_model.pb"}));
}
#endif

template <>
std::basic_string<char> get_saved_model_name<char>() {
    return "/saved_model.pb";
}
template <>
std::basic_string<char> get_variables_index_name<char>() {
    return "/variables/variables.index";
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_saved_model_name<wchar_t>() {
    return L"/saved_model.pb";
}
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>() {
    return L"/variables/variables.index";
}
#endif

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
