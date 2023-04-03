// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "graph_iterator_proto.hpp"
#include "openvino/util/file_util.hpp"
#include "saved_model.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

struct VIBlock;

template <typename T>
std::basic_string<T> get_saved_model_name() {}
template <typename T>
std::basic_string<T> get_variables_index_name() {}

template <>
std::basic_string<char> get_saved_model_name<char>();
template <>
std::basic_string<char> get_variables_index_name<char>();

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_saved_model_name<wchar_t>();
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>();
#endif

// Stores information about variables index
class SavedModelVariablesIndex {
    // Contains maximum amount of shards, used for creating corrext extension
    int32_t m_total_shards;
    // Contains BundleEntryProto variables list, readed from .index file
    std::map<std::string, std::vector<char>> m_variables_index;
    // List of opened data files for using with BundleEntryProto
    std::map<int32_t, std::shared_ptr<std::ifstream>> m_data_files;
    // List of mapped variables which could be read using TrackableObjectGraph
    std::map<std::string, std::string> m_variables_map;

public:
    /// \brief Reads variables from opened variable index file. Can cause an asserts in case of issues.
    /// \param vi_stream Opened stream file, file pointer doesn't matter, it will be rewind internally.
    /// \param path A path to file with variables data
    /// \returns Returns true in case of everything loads successfully, false otherwise
    bool read_variables(std::ifstream& vi_stream, const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    /// \brief Reads variables from opened variable index file. Can cause an asserts in case of issues.
    /// \param vi_stream Opened stream file, file pointer doesn't matter, it will be rewind internally.
    /// \param path A path to file with variables data
    /// \returns Returns true in case of everything loads successfully, false otherwise
    bool read_variables(std::ifstream& vi_stream, const std::wstring& path);
#endif

    /// \brief Returns data and size of data of stored variable
    /// \param name Name of variable
    /// \param data Pointer on a pointer where data pointer will be returned
    /// \param size Pointer on a variable which will stores data size
    /// \returns Returns true in case variable was found, false otherwise (data and size will be untouched)
    bool get_variable(const std::string& name, const char** data, size_t* size) const {
        auto varItem = m_variables_index.find(name);
        if (varItem == m_variables_index.end()) {
            return false;
        }
        if (data != nullptr) {
            *data = varItem->second.data();
        }
        if (size != nullptr) {
            *size = varItem->second.size();
        }
        return true;
    }

    /// \brief Returns data and size of data of mapped variable from trackable object graph to variables index
    /// \param name Name of a mapping variable
    /// \param data Pointer on a pointer where data pointer will be returned
    /// \param size Pointer on a variable which will stores data size
    /// \returns Returns true in case variable was found, false otherwise (data and size will be untouched)
    bool get_mapped_variable(const std::string& name, const char** data, size_t* size) const {
        auto mapItem = m_variables_map.find(name);
        if (mapItem == m_variables_map.end()) {
            return false;
        }
        return get_variable(mapItem->second, data, size);
    }

    /// \brief Checks if variable has a mapped pair
    /// \param name Name of variable for checking existance
    /// \returns True in case variable has mapped value and false otherwise
    bool has_mapped_variable(const std::string& name) const {
        auto mapItem = m_variables_map.find(name);
        return mapItem != m_variables_map.end();
    }

    /// \brief Returns shared pointer to a requested shard_id, or nullptr in case of shard_id isn't found
    /// \param shard_id Requested shard_id
    /// \returns Valid shared_ptr with ifstream or with nullptr if shard isn't found
    std::shared_ptr<std::ifstream> get_data_file(const int32_t shard_id) const {
        auto result = m_data_files.find(shard_id);
        return result != m_data_files.end() ? result->second : nullptr;
    }

    /// \brief Adds variable mapping to the variables map
    /// \param var_name Variable full name (from .index file)
    /// \param map_name Mapped name
    /// \param rewrite Rewrite mapped value in case it exists
    /// \returns True if map updated. False if nothing changed (if variable exists and rewrite is false).
    bool map_variable(const std::string& var_name, const std::string& map_name, bool rewrite = false) {
        if (m_variables_map.find(var_name) != m_variables_map.end() && rewrite == false) {
            return false;
        }

        m_variables_map[var_name] = map_name;
        return true;
    }

private:
    /// \brief Reads block structure of .index file
    /// \param[in,out] fs Filestream of .index file, position in file will be updated
    /// \param[in] index Variables index block which stores information about block
    /// \param[out] data Block data will be readed
    /// \param[out] offset Offset of block start
    /// \param[out] offset_end Offset of block end
    void read_variables_index_block(std::ifstream& fs,
                                    const VIBlock& index,
                                    std::vector<char>& data,
                                    uint32_t& offset,
                                    uint32_t& offset_end);
    /// \brief Reads key=value pair from provided pointer
    /// \param[in,out] ptr Actual pointer, will be moved to the end of readed pair (to read next)
    /// \param[in] ptr_end End of memory which shouldn't be passed in case of broken structure
    /// \param[out] key Key name
    /// \param[out] value Stored value for key (isn't a pure string, data block)
    /// \param[out] val_lenght Length of readed value
    void read_variables_index_pair(char*& ptr,
                                   const char* ptr_end,
                                   std::string& key,
                                   char*& value,
                                   uint32_t& val_length);
    /// \brief Reads .index file and stores key=value map in provided varIndex
    /// \param[in,out] fs Filestream should be parsed. Position in file will be updated
    /// \param[out] varIndex Variables indx (key=value) from given filestream
    void read_variables_index(std::ifstream& fs, std::map<std::string, std::vector<char>>& varIndex);
    /// \brief Reads bundle header if it is available. Checks version and saves info about amount of shards
    void read_bundle_header();
    /// \brief Reads key=value map from storef _CHECKPOINTABLE_OBJECT_GRAPH variable
    void read_checkpointable_object_graph();
};

// Loads graph from Tensorflow Saved Model file (saved_model.pb)
class GraphIteratorSavedModel : public GraphIteratorProto {
    std::shared_ptr<::tensorflow::SavedModel> m_saved_model;
    std::shared_ptr<SavedModelVariablesIndex> m_variables_index;
    std::shared_ptr<std::map<std::string, std::string>> m_inputs_map;
    std::shared_ptr<std::map<std::string, std::string>> m_outputs_map;

public:
    template <typename T>
    GraphIteratorSavedModel(const std::basic_string<T>& path, const std::string& tags)
        : m_saved_model(std::make_shared<::tensorflow::SavedModel>()) {
        this->read_saved_model(path, tags);
    }

    static bool is_supported(const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    static bool is_supported(const std::wstring& path);
#endif

    std::shared_ptr<SavedModelVariablesIndex> get_variables_index() {
        return m_variables_index;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_saved_model_input_names() const {
        return m_inputs_map;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_saved_model_output_names() const {
        return m_outputs_map;
    }

private:
    bool is_valid_signature(const ::tensorflow::SignatureDef& signature) const;

    template <typename T>
    bool read_saved_model(const std::basic_string<T>& path, const std::string& tags) {
        std::ifstream sm_stream{path + get_saved_model_name<T>(), std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(sm_stream && sm_stream.is_open(), "Model file does not exist");

        std::basic_string<T> varIndexPath = path + get_variables_index_name<T>();
        if (ov::util::file_exists(varIndexPath)) {
            m_variables_index = std::make_shared<SavedModelVariablesIndex>();
            std::ifstream vi_stream{varIndexPath, std::ifstream::in | std::ifstream::binary};
            FRONT_END_GENERAL_CHECK(vi_stream && vi_stream.is_open(),
                                    "Saved Model's variable index file does not exist");
            FRONT_END_GENERAL_CHECK(m_variables_index->read_variables(vi_stream, path),
                                    "Saved Model's variable index file cannot be parsed");
        }

        bool res = m_saved_model->ParseFromIstream(&sm_stream);
        FRONT_END_GENERAL_CHECK(res && m_saved_model->meta_graphs_size(), "Saved Model cannot be parsed");

        for (const auto& meta_graph : m_saved_model->meta_graphs()) {
            if (!meta_graph.has_graph_def()) {
                continue;
            }

            if (m_saved_model->meta_graphs_size() > 1) {
                bool tag_found = false;
                for (const auto& tag : meta_graph.meta_info_def().tags()) {
                    if (tags.find(tag) != std::string::npos) {
                        tag_found = true;
                        break;
                    }
                }
                if (!tag_found) {
                    continue;
                }
            }

            std::map<std::string, const ::tensorflow::SignatureDef*> validSignatures = {};
            for (const auto& sit : meta_graph.signature_def()) {
                const std::string& key = sit.first;
                const ::tensorflow::SignatureDef& val = sit.second;
                if (is_valid_signature(val)) {
                    validSignatures[key] = &val;
                }
            }

            auto serving_default = validSignatures.find("serving_default");

            if (serving_default != validSignatures.end()) {
                m_inputs_map = std::make_shared<std::map<std::string, std::string>>();
                m_outputs_map = std::make_shared<std::map<std::string, std::string>>();
                for (const auto& input : serving_default->second->inputs()) {
                    (*m_inputs_map)[input.second.name()] = input.first;
                }
                for (const auto& output : serving_default->second->outputs()) {
                    (*m_outputs_map)[output.second.name()] = output.first;
                }
            }

            m_graph_def = std::make_shared<::tensorflow::GraphDef>(meta_graph.graph_def());

            // Update variables map using information by resolving AssignVariableOp graph nodes
            std::map<std::string, std::string> var_map;
            map_assignvariable(m_graph_def, var_map);
            for (auto var : var_map) {
                m_variables_index->map_variable(var.first, var.second);
            }

            auto nodes_size = m_graph_def->node_size();
            m_decoders.resize(static_cast<size_t>(nodes_size));
            for (int node_ind = 0; node_ind < nodes_size; ++node_ind) {
                m_decoders[node_ind] = std::make_shared<DecoderProto>(&m_graph_def->node(node_ind), m_graph_def);
            }

            // initialize a library map
            auto num_funcs = m_graph_def->library().function_size();
            for (int func_ind = 0; func_ind < num_funcs; ++func_ind) {
                auto func = m_graph_def->library().function(func_ind);
                auto func_name = func.signature().name();
                m_library_map.insert(std::pair<std::string, int>(func_name, func_ind));
            }

            return true;
        }

        FRONT_END_GENERAL_CHECK(false, "Saved Model doesn't contain MetaGraph with requested tag");

        return false;
    }

    /// \brief Reads relationship between VarHandleOp - RestoreV2 - AssignVariableOp and
    /// stores this information in a provided key=value map. Where key - name of VarHandleOp,
    /// value - long variable name which is stored in RestoreV2.
    /// It needs to map VarHandleOp to right place in .index file.
    /// \param[in] graph_def GraphDef object for analysis
    /// \param[out] variables_map Map of variables found in graph_def
    void map_assignvariable(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                            std::map<std::string, std::string>& variables_map) const;
};  // GraphIteratorSavedModel

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
