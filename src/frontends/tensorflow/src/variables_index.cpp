// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <fstream>
#include <string>

#include "checkpoint_utils.hpp"
#include "graph_iterator_saved_model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/mmap_object.hpp"
#include "ov_tensorflow/tensor_bundle.pb.h"
#include "ov_tensorflow/trackable_object_graph.pb.h"
#include "tf_utils.hpp"

#ifdef ENABLE_SNAPPY_COMPRESSION
#    include "snappy.h"
#endif

namespace ov {
namespace frontend {
namespace tensorflow {

void VariablesIndex::read_variables_index_block(std::ifstream& fs,
                                                const VIBlock& index,
                                                std::vector<char>& data,
                                                uint32_t& offset,
                                                uint32_t& offset_end) {
    size_t block_size = index.m_size;
    data.clear();
    data.resize(block_size + BLOCK_TRAILER_SIZE);
    FRONT_END_GENERAL_CHECK(index.m_offset <= m_variables_index_size,
                            "Block offset is bigger than variables index size");
    FRONT_END_GENERAL_CHECK(index.m_offset + data.size() <= m_variables_index_size,
                            "Block size is bigger than variables index size");
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
    uint32_t numRestarts = decode_fixed32(data.data() + block_size - sizeof(uint32_t));
    size_t maxRestarts = (block_size - sizeof(uint32_t)) / sizeof(uint32_t);
    FRONT_END_GENERAL_CHECK(maxRestarts >= numRestarts, "Wrong restarts value");
    offset_end = static_cast<uint32_t>(block_size) - ((numRestarts + 1) * sizeof(uint32_t));
    offset = decode_fixed32(data.data() + offset_end);
}

void VariablesIndex::read_variables_index_pair(char*& ptr,
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

void VariablesIndex::read_variables_index(std::ifstream& fs, std::map<std::string, std::vector<char>>& varIndex) {
    fs.seekg(0, std::ios::end);
    m_variables_index_size = fs.tellg();

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

void VariablesIndex::read_bundle_header() {
    auto item = m_variables_index.find("");
    FRONT_END_GENERAL_CHECK(item != m_variables_index.end(), "Bundle Header isn't found in index");

    ::tensorflow::BundleHeaderProto bundleHeader;
    FRONT_END_GENERAL_CHECK(bundleHeader.ParseFromArray(item->second.data(), static_cast<int>(item->second.size())),
                            "Bundle Header: Cannot parse Bundle Header");
    FRONT_END_GENERAL_CHECK(bundleHeader.version().producer() == 1, "Bundle Header: Unsupported producer version");
    FRONT_END_GENERAL_CHECK(bundleHeader.version().min_consumer() == 0, "Bundle Header: Unsupported consumer version");
    FRONT_END_GENERAL_CHECK(bundleHeader.endianness() == 0, "Bundle Header: BIG endian isn't supported");

    m_total_shards = bundleHeader.num_shards();
}

void VariablesIndex::read_checkpointable_object_graph() {
    m_variables_map.clear();

    auto item = m_variables_index.find("_CHECKPOINTABLE_OBJECT_GRAPH");
    if (item == m_variables_index.end()) {
        // Might be missing for some models. In such case all variables should be resolved thru RestoreV2
        return;
    }

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
    if (m_mmap_enabled) {
        auto srcPtr = static_cast<char*>(shard->second.mmap->data() + entry.offset() + chg);
        std::copy(srcPtr, srcPtr + entry.size() - chg, data.data());
    } else {
        shard->second.stream->seekg(entry.offset() + chg);
        shard->second.stream->read(data.data(), entry.size() - chg);
    }

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

bool VariablesIndex::read_variables(std::ifstream& vi_stream, const std::string& path, const bool is_saved_model) {
    m_variables_index.clear();
    read_variables_index(vi_stream, m_variables_index);
    read_bundle_header();

    std::vector<char> suffix(20);
    for (int32_t shard = 0; shard < m_total_shards; ++shard) {
        std::snprintf(suffix.data(), suffix.size(), "data-%05d-of-%05d", shard, m_total_shards);
        std::string fullPath;
        if (is_saved_model) {
            fullPath = ov::util::path_join({path, "variables", std::string("variables.") + suffix.data()});
        } else {
            fullPath = path + "." + suffix.data();
        }
        if (m_mmap_enabled) {
            m_data_files[shard].mmap = load_mmap_object(fullPath);
            FRONT_END_GENERAL_CHECK(m_data_files[shard].mmap->data(), "Variable index data cannot be mapped");
        } else {
            m_data_files[shard].stream = std::shared_ptr<std::ifstream>(
                new std::ifstream(fullPath.c_str(), std::ifstream::in | std::ifstream::binary));
            FRONT_END_GENERAL_CHECK(m_data_files[shard].stream->is_open(), "Variable index data file does not exist");
        }
    }

    read_checkpointable_object_graph();
    return true;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool VariablesIndex::read_variables(std::ifstream& vi_stream, const std::wstring& path, const bool is_saved_model) {
    m_variables_index.clear();
    read_variables_index(vi_stream, m_variables_index);
    read_bundle_header();

    std::vector<wchar_t> suffix(20);
    for (int32_t shard = 0; shard < m_total_shards; ++shard) {
        swprintf_s(suffix.data(), suffix.size(), L"data-%05d-of-%05d", shard, m_total_shards);
        std::wstring fullPath;
        if (is_saved_model) {
            fullPath = ov::util::path_join_w({path, L"variables", std::wstring(L"variables.") + suffix.data()});
        } else {
            fullPath = path + L"." + suffix.data();
        }
        if (m_mmap_enabled) {
            m_data_files[shard].mmap = load_mmap_object(fullPath);
            FRONT_END_GENERAL_CHECK(m_data_files[shard].mmap->data(), "Variable index data cannot be mapped");
        } else {
            m_data_files[shard].stream = std::shared_ptr<std::ifstream>(
                new std::ifstream(fullPath.c_str(), std::ifstream::in | std::ifstream::binary));
            FRONT_END_GENERAL_CHECK(m_data_files[shard].stream->is_open(), "Variable index data file does not exist");
        }
    }

    read_checkpointable_object_graph();
    return true;
}
#endif

struct PtrNode {
    using SharedPtrNode = std::shared_ptr<PtrNode>;

    const ::tensorflow::NodeDef* node;
    std::vector<SharedPtrNode> inputs;
    std::vector<SharedPtrNode> outputs;

    PtrNode() : node(nullptr), inputs(), outputs() {}

    PtrNode(const ::tensorflow::NodeDef& src_node) {
        node = &src_node;
    }

    void associate_node(const SharedPtrNode shared_node, const std::map<std::string, SharedPtrNode>& node_dictionary) {
        FRONT_END_GENERAL_CHECK(shared_node.get() == this, "Only current object is expected for association");
        std::vector<std::string> parsedName;
        for (const auto& input_name : node->input()) {
            parse_node_name(input_name, parsedName);

            auto input_node = node_dictionary.find(parsedName[0]);
            if (input_node == node_dictionary.end()) {
                continue;
            }

            input_node->second->outputs.push_back(shared_node);
            inputs.push_back(input_node->second);
        }
    }

    void find_parent_by_op(const std::string& op,
                           std::vector<SharedPtrNode>& result,
                           std::shared_ptr<std::vector<const PtrNode*>> walked = nullptr) const {
        if (walked.get() == nullptr) {
            walked = std::make_shared<std::vector<const PtrNode*>>();
        }
        for (auto input : inputs) {
            if (input->op() == op) {
                result.push_back(input);
            }
            if (find(walked->begin(), walked->end(), input.get()) == walked->end()) {
                walked->push_back(this);
                input->find_parent_by_op(op, result, walked);
            }
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
                                           std::map<std::string, PtrNode::SharedPtrNode>& node_dictionary) {
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

    std::map<std::string, PtrNode::SharedPtrNode> nodes;

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
        auto shared_node = std::make_shared<PtrNode>(node);
        shared_node->associate_node(shared_node, nodes);
        nodes[node.name()] = shared_node;

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

void VariablesIndex::map_assignvariable(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                                        std::map<std::string, std::string>& variables_map,
                                        HashTableKeysValuesMap& hash_table_keys_map,
                                        HashTableKeysValuesMap& hash_table_values_map) {
    std::map<std::string, PtrNode::SharedPtrNode> nodes;

    for (const auto& node : graph_def->node()) {
        auto shared_node = std::make_shared<PtrNode>(node);
        shared_node->associate_node(shared_node, nodes);
        nodes[node.name()] = shared_node;

        if (node.op() == "StatefulPartitionedCall") {
            read_stateful_partitioned_call(graph_def, node, nodes);
        }
    }

    for (const auto& node : nodes) {
        if (node.second->op() == "AssignVariableOp") {
            // TODO: assets reading

            std::vector<PtrNode::SharedPtrNode> restorev2_nodes;
            std::vector<PtrNode::SharedPtrNode> varhandle_nodes;

            node.second->find_parent_by_op("RestoreV2", restorev2_nodes);
            node.second->find_parent_by_op("VarHandleOp", varhandle_nodes);

            if (restorev2_nodes.size() == 1 && varhandle_nodes.size() == 1) {
                std::vector<std::string> restore_output;

                FRONT_END_GENERAL_CHECK(node.second->inputs.size() >= 2,
                                        "Amount of AssignVariableOp inputs is less than expected");
                // Here is known ways to find a correct RestoreV2 output index:
                if (node.second->inputs[1]->inputs.size() >= 1 &&
                    node.second->inputs[1]->inputs[0]->node->op() == "RestoreV2") {
                    // Expected path is: RestoreV2 -(output_index)-(0)-> AnyNode -(0)-(1)-> AssignVariableOp
                    PtrNode::parse_node_name(node.second->inputs[1]->node->input(0), restore_output);
                } else if (node.second->inputs[1]->node->op() == "RestoreV2" && node.second->node->input_size() >= 2) {
                    // Expected path is: RestoreV2 -(output_index)-(1)-> AssignVariableOp
                    PtrNode::parse_node_name(node.second->node->input(1), restore_output);
                } else {
                    FRONT_END_THROW("Unexpected topology near AssignVariableOp");
                }

                int output_index = std::atoi(restore_output[restore_output.size() - 1].c_str());

                // Expected path is: Const(tensor_names) -(0)-(1)-> RestoreV2
                const auto& variable_name =
                    restorev2_nodes[0]->inputs[1]->node->attr().at("value").tensor().string_val(output_index);

                variables_map[varhandle_nodes[0]->node->name()] = variable_name;
            }
        } else if (node.second->op() == "Assign") {
            std::vector<PtrNode::SharedPtrNode> restorev2_nodes;
            std::vector<PtrNode::SharedPtrNode> variablev2_nodes;

            node.second->find_parent_by_op("RestoreV2", restorev2_nodes);
            node.second->find_parent_by_op("VariableV2", variablev2_nodes);

            // Added support of Variable nodes in case no associated VariableV2 nodes found
            if (variablev2_nodes.size() == 0) {
                node.second->find_parent_by_op("Variable", variablev2_nodes);
            }

            if (restorev2_nodes.size() == 1 && variablev2_nodes.size() == 1) {
                std::vector<std::string> restore_output;

                FRONT_END_GENERAL_CHECK(node.second->node->input_size() >= 2,
                                        "Amount of Assign inputs is less than expected");
                // Expected path is: RestoreV2 -(output_index)-(1)-> Assign
                PtrNode::parse_node_name(node.second->node->input(1), restore_output);

                int output_index = std::atoi(restore_output[restore_output.size() - 1].c_str());

                // Expected path is: Const(tensor_names) -(0)-(1)-> RestoreV2
                const auto& variable_name =
                    restorev2_nodes[0]->inputs[1]->node->attr().at("value").tensor().string_val(output_index);

                variables_map[variablev2_nodes[0]->node->name()] = variable_name;
            }
        } else if (node.second->op() == "LookupTableImportV2") {
            std::vector<PtrNode::SharedPtrNode> hash_tablev2_nodes;
            node.second->find_parent_by_op("HashTableV2", hash_tablev2_nodes);
            if (hash_tablev2_nodes.size() == 0 || node.second->node->input_size() < 3) {
                continue;
            }

            // extract tensors with keys and values
            // expect Constant (with keys) -> LookupTableImportV2 and Constant (with values) -> LookupTableImportV2
            if (node.second->inputs[1]->node->op() != "Const" || node.second->inputs[2]->node->op() != "Const") {
                continue;
            }

            auto hash_tablev2_name = hash_tablev2_nodes[0]->node->name();
            auto ov_tensor_keys =
                unpack_tensor_proto(node.second->inputs[1]->node->attr().at("value").tensor()).as<ov::Tensor>();
            auto ov_tensor_values =
                unpack_tensor_proto(node.second->inputs[2]->node->attr().at("value").tensor()).as<ov::Tensor>();

            // create Constant nodes for keys and values and store them in maps
            // these Constant nodes can be retrieved during conversion stage for conversion HashTableV2 operation
            auto keys_const = std::make_shared<ov::op::v0::Constant>(ov_tensor_keys);
            auto values_const = std::make_shared<ov::op::v0::Constant>(ov_tensor_values);

            hash_table_keys_map[hash_tablev2_name] = keys_const;
            hash_table_values_map[hash_tablev2_name] = values_const;
        }
    }

    // Removing cross-links, otherwise memory leak will be caused by lost shared pointers
    for (auto node : nodes) {
        node.second->inputs.clear();
        node.second->outputs.clear();
    }

    nodes.clear();
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
