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
std::basic_string<T> getSMName() {}
template <typename T>
std::basic_string<T> getVIName() {}

template <>
std::basic_string<char> getSMName<char>();
template <>
std::basic_string<char> getVIName<char>();

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> getSMName<wchar_t>();
template <>
std::basic_string<wchar_t> getVIName<wchar_t>();
#endif

// Stores information about variables index
class SMVariablesIndex {
    int32_t totalShards;
    std::map<std::string, std::vector<char>> varIndex;
    std::map<int32_t, std::shared_ptr<std::ifstream>> dataFiles;
    std::map<std::string, std::string> varMap;

public:
    /// \brief Reads variables from opened variable index file. Can cause an asserts in case of issues.
    /// \param vi_stream Opened stream file, file pointer doesn't matter, it will be rewind internally.
    /// \param path A path to file with variables data
    /// \returns Returns true in case of everything loads successfully, false otherwise
    bool readVariables(std::ifstream& vi_stream, const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    /// \brief Reads variables from opened variable index file. Can cause an asserts in case of issues.
    /// \param vi_stream Opened stream file, file pointer doesn't matter, it will be rewind internally.
    /// \param path A path to file with variables data
    /// \returns Returns true in case of everything loads successfully, false otherwise
    bool readVariables(std::ifstream& vi_stream, const std::wstring& path);
#endif

    /// \brief Returns data and size of data of stored variable
    /// \param name Name of variable
    /// \param data Pointer on a pointer where data pointer will be returned
    /// \param size Pointer on a variable which will stores data size
    /// \returns Returns true in case variable was found, false otherwise (data and size will be untouched)
    bool getVariable(const std::string& name, char** data, size_t* size) {
        auto varItem = varIndex.find(name);
        if (varItem == varIndex.end()) {
            return false;
        }
        *data = varItem->second.data();
        *size = varItem->second.size();
        return true;
    }

    /// \brief Returns data and size of data of mapped variable from trackable object graph to variables index
    /// \param name Name of a mapping variable
    /// \param data Pointer on a pointer where data pointer will be returned
    /// \param size Pointer on a variable which will stores data size
    /// \returns Returns true in case variable was found, false otherwise (data and size will be untouched)
    bool getMappedVariable(const std::string& name, char** data, size_t* size) {
        auto mapItem = varMap.find(name);
        if (mapItem == varMap.end()) {
            return false;
        }
        return getVariable(mapItem->second, data, size);
    }

    std::shared_ptr<std::ifstream> getDataFile(const int32_t shard_id) {
        auto result = dataFiles.find(shard_id);
        return result != dataFiles.end() ? result->second : nullptr;
    }

    bool mapVariable(const std::string var_name, const std::string map_name, bool rewrite = false) {
        if (varMap.find(var_name) != varMap.end() && rewrite == false) {
            return false;
        }

        varMap[var_name] = map_name;
        return true;
    }

private:
    // Internal implementation of saved model reading
    void readVIBlock(std::ifstream& fs,
                     const VIBlock* index,
                     std::vector<char>& data,
                     uint32_t* offset,
                     uint32_t* offset_end);
    void readVIPair(char** ptr, const char* ptr_end, std::string& key, char** value, uint32_t* val_length);
    void readVarIndex(std::ifstream& fs, std::map<std::string, std::vector<char>>& varIndex);
    void readBundleHeader();
    void readCMOGraph();
};

// Loads graph from Tensorflow Saved Model file (saved_model.pb)
class GraphIteratorSavedModel : public GraphIteratorProto {
    std::shared_ptr<::tensorflow::SavedModel> m_saved_model;
    std::shared_ptr<SMVariablesIndex> m_variables_index;

public:
    template <typename T>
    GraphIteratorSavedModel(const std::basic_string<T>& path)
        : m_saved_model(std::make_shared<::tensorflow::SavedModel>()) {
        this->readSavedModel(path);
    }

    static bool isSavedModel(const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    static bool isSavedModel(const std::wstring& path);
#endif

    std::shared_ptr<SMVariablesIndex> getVariablesIndex() {
        return m_variables_index;
    }

private:
    bool isValidSignature(const ::tensorflow::SignatureDef& signature);

    template <typename T>
    bool readSavedModel(const std::basic_string<T>& path) {
        std::ifstream sm_stream{path + getSMName<T>(), std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(sm_stream && sm_stream.is_open(), "Model file does not exist");

        std::basic_string<T> varIndexPath = path + getVIName<T>();
        if (ov::util::file_exists(varIndexPath)) {
            m_variables_index = std::make_shared<SMVariablesIndex>();
            std::ifstream vi_stream{varIndexPath, std::ifstream::in | std::ifstream::binary};
            FRONT_END_GENERAL_CHECK(vi_stream && vi_stream.is_open(),
                                    "Saved Model's variable index file does not exist");
            FRONT_END_GENERAL_CHECK(m_variables_index->readVariables(vi_stream, path),
                                    "Saved Model's variable index file cannot be parsed");
        }

        bool res = m_saved_model->ParseFromIstream(&sm_stream);
        FRONT_END_GENERAL_CHECK(res && m_saved_model->meta_graphs_size(), "Saved Model cannot be parsed");

        // Supported only first meta_graph at the moment
        const ::tensorflow::MetaGraphDef& meta_graph = m_saved_model->meta_graphs(0);
        FRONT_END_GENERAL_CHECK(meta_graph.has_graph_def(), "Saved Model doesn't contain GraphDef");

        std::vector<std::string> validSignatures = {};
        for (auto& sit : meta_graph.signature_def()) {
            const std::string& key = sit.first;
            const ::tensorflow::SignatureDef& val = sit.second;
            if (isValidSignature(val)) {
                validSignatures.push_back(key);
            } else {
                OPENVINO_ASSERT(false, "Saved Model contains invalid signatures");
            }
        }

        // TODO: assets reading

        m_graph_def = std::make_shared<::tensorflow::GraphDef>(meta_graph.graph_def());

        // TODO: update loading nodes by using data from separately
        auto nodes_size = m_graph_def->node_size();
        m_decoders.resize(static_cast<size_t>(nodes_size));
        for (int node_ind = 0; node_ind < nodes_size; ++node_ind) {
            m_decoders[node_ind] = std::make_shared<DecoderProto>(&m_graph_def->node(node_ind));
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
};  // GraphIteratorSavedModel

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
