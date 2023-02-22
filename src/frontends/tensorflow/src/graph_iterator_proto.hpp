// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <map>
#include <vector>

#include "decoder_argdef.hpp"
#include "decoder_proto.hpp"
#include "graph.pb.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/util/file_util.hpp"
#include "saved_model.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

class GraphIteratorProto : public GraphIterator {
protected:
    std::shared_ptr<::tensorflow::GraphDef> m_graph_def;
    std::shared_ptr<::tensorflow::FunctionDef> m_func_def;

    size_t node_index = 0;
    std::vector<std::shared_ptr<DecoderBase>> m_decoders;
    std::unordered_map<std::string, int> m_library_map;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;

    GraphIteratorProto()
        : m_graph_def(std::make_shared<::tensorflow::GraphDef>()),
          m_func_def(nullptr),
          m_library_map() {}

public:
    GraphIteratorProto(const std::shared_ptr<::tensorflow::GraphDef>& graph_def,
                       const std::shared_ptr<::tensorflow::FunctionDef>& func_def,
                       const std::unordered_map<std::string, int>& library_map)
        : m_graph_def(graph_def),
          m_func_def(func_def),
          m_library_map(library_map) {
        auto nodes_size = m_func_def->node_def_size();
        auto input_size = m_func_def->signature().input_arg_size();
        auto output_size = m_func_def->signature().output_arg_size();
        auto ret_map = m_func_def->ret();

        // fill all inputs from library functions
        // these input_arg objects are of different type OpDef_ArgDef
        // they are not NodeDef so we use separate Decoder class
        for (int input_ind = 0; input_ind < input_size; ++input_ind) {
            auto input_arg = &m_func_def->signature().input_arg(input_ind);
            m_input_names.push_back(input_arg->name());
            m_decoders.push_back(std::make_shared<DecoderArgDef>(input_arg, m_graph_def, m_func_def, "input_arg"));
        }

        // fill all node defs from library functions
        for (int node_ind = 0; node_ind < nodes_size; ++node_ind) {
            m_decoders.push_back(
                std::make_shared<DecoderProto>(&(m_func_def->node_def(node_ind)), m_graph_def, m_func_def));
        }

        // fill all outputs from library functions
        // these output_arg objects are of different type OpDef_ArgDef
        // they are not NodeDef so we use separate Decoder class
        for (int output_ind = 0; output_ind < output_size; ++output_ind) {
            auto output_arg = &m_func_def->signature().output_arg(output_ind);
            m_output_names.push_back(output_arg->name());
            auto producer_name = ret_map.at(output_arg->name());
            m_decoders.push_back(
                std::make_shared<DecoderArgDef>(output_arg, m_graph_def, m_func_def, "output_arg", producer_name));
        }
    }

    template <typename T>
    GraphIteratorProto(const std::basic_string<T>& path)
        : m_graph_def(std::make_shared<::tensorflow::GraphDef>()),
          m_func_def(nullptr) {
        std::ifstream pb_stream(path, std::ios::in | std::ifstream::binary);

        FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file does not exist");
        FRONT_END_GENERAL_CHECK(m_graph_def->ParseFromIstream(&pb_stream), "Model cannot be parsed");

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
    }

    /// \brief Check if the input file is supported
    template <typename T>
    static bool is_supported(const std::basic_string<T>& path) {
        std::ifstream pb_stream(path, std::ios::in | std::ifstream::binary);
        auto graph_def = std::make_shared<::tensorflow::GraphDef>();
        return pb_stream && pb_stream.is_open() && graph_def->ParsePartialFromIstream(&pb_stream);
    }

    /// \brief Set iterator to the start position
    void reset() override {
        node_index = 0;
    }

    /// \brief Return a number of nodes in the graph
    size_t size() const override {
        return m_decoders.size();
    }

    /// \brief Move to the next node in the graph
    void next() override {
        node_index++;
    }

    /// \brief Check if the graph is fully traversed
    bool is_end() const override {
        return node_index >= m_decoders.size();
    }

    /// \brief Return NodeContext for the current node that iterator points to
    std::shared_ptr<DecoderBase> get_decoder() const override {
        return m_decoders[node_index];
    }

    /// \brief Get GraphIterator for library funnction by name
    std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const override {
        if (m_library_map.count(func_name)) {
            auto func_ind = m_library_map.at(func_name);
            auto func_size = m_graph_def->library().function_size();
            FRONT_END_GENERAL_CHECK(
                0 <= func_ind && func_ind < func_size,
                "[TensorFlow Error] Internal Error: incorrect library map to cache function indices by names.");

            auto func = m_graph_def->library().function(func_ind);
            auto func_ptr = std::make_shared<::tensorflow::FunctionDef>(func);
            return std::make_shared<GraphIteratorProto>(m_graph_def, func_ptr, m_library_map);
        }

        return nullptr;
    }

    /// \brief Get input names in the original order. Used for the library functions
    std::vector<std::string> get_input_names() const override {
        return m_input_names;
    }

    /// \brief Get output names in the original order. Used for the library functions
    std::vector<std::string> get_output_names() const override {
        return m_output_names;
    }
};

struct SMBlock;

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

// Loads graph from Tensorflow Saved Model file (saved_model.pb)
class GraphIteratorSavedModel : public GraphIteratorProto {
    std::shared_ptr<::tensorflow::SavedModel> m_saved_model;
    std::map<std::string, std::vector<char>> varIndex;
    int32_t totalShards;
    std::map<int32_t, std::shared_ptr<std::ifstream>> dataFiles;
    std::map<std::string, std::string> varMap;

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

private:
    bool isValidSignature(const ::tensorflow::SignatureDef& signature);
    bool readVariables(std::ifstream& vi_stream, const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    bool readVariables(std::ifstream& vi_stream, const std::wstring& path);
#endif

    template <typename T>
    bool readSavedModel(const std::basic_string<T>& path) {
        std::ifstream sm_stream{path + getSMName<T>(), std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(sm_stream && sm_stream.is_open(), "Model file does not exist");

        std::basic_string<T> varIndexPath = path + getVIName<T>();
        if (ov::util::file_exists(varIndexPath)) {
            std::ifstream vi_stream{varIndexPath, std::ifstream::in | std::ifstream::binary};
            FRONT_END_GENERAL_CHECK(vi_stream && vi_stream.is_open(),
                                    "Saved Model's variable index file does not exist");
            FRONT_END_GENERAL_CHECK(readVariables(vi_stream, path),
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

    // Internal implementation of saved model reading
    void readSMBlock(std::ifstream& fs,
                     const SMBlock* index,
                     std::vector<char>& data,
                     uint32_t* offset,
                     uint32_t* offset_end);
    void readSMPair(char** ptr, const char* ptr_end, std::string& key, char** value, uint32_t* val_length);
    void readVarIndex(std::ifstream& fs, std::map<std::string, std::vector<char>>& varIndex);
    void readBundleHeader();
    void readCMOGraph();
};  // GraphIteratorSavedModel

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
