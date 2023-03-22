// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "graph_iterator_proto.hpp"
#include "openvino/util/file_util.hpp"
#include "variables_index.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

template <typename T>
std::basic_string<T> get_variables_index_name(const std::basic_string<T> name) {}

template <>
std::basic_string<char> get_variables_index_name<char>(const std::string name);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>(const std::wstring name);
#endif

// Loads graph from Tensorflow MetaGraph file (*.meta)
class GraphIteratorMeta : public GraphIteratorProto {
    std::shared_ptr<::tensorflow::MetaGraphDef> m_metagraph_def;
    std::shared_ptr<VariablesIndex> m_variables_index;
    std::shared_ptr<std::map<std::string, std::string>> m_inputs_map;
    std::shared_ptr<std::map<std::string, std::string>> m_outputs_map;

public:
    template <typename T>
    GraphIteratorMeta(const std::basic_string<T>& path)
        : m_metagraph_def(std::make_shared<::tensorflow::MetaGraphDef>()) {
        this->read_meta(path);
    }

    static bool is_supported(const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    static bool is_supported(const std::wstring& path);
#endif

    std::shared_ptr<VariablesIndex> get_variables_index() {
        return m_variables_index;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_metagraph_input_names() const {
        return m_inputs_map;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_metagraph_output_names() const {
        return m_outputs_map;
    }

private:
    bool is_valid_signature(const ::tensorflow::SignatureDef& signature) const;

    template <typename T>
    bool read_meta(const std::basic_string<T>& path) {
        std::basic_string<T> model_path = path.substr(0, path.find_last_of('.'));

        std::ifstream mg_stream{path, std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(mg_stream && mg_stream.is_open(), "Model file does not exist");

        std::basic_string<T> varIndexPath = get_variables_index_name<T>(model_path);
        if (ov::util::file_exists(varIndexPath)) {
            m_variables_index = std::make_shared<VariablesIndex>();
            std::ifstream vi_stream{varIndexPath, std::ifstream::in | std::ifstream::binary};
            FRONT_END_GENERAL_CHECK(vi_stream && vi_stream.is_open(), "MetaGraph's variable index file does not exist");
            FRONT_END_GENERAL_CHECK(m_variables_index->read_variables(vi_stream, model_path, false),
                                    "MetaGraph's variable index file cannot be parsed");
        }

        bool res = m_metagraph_def->ParseFromIstream(&mg_stream);
        FRONT_END_GENERAL_CHECK(res && m_metagraph_def->has_graph_def(), "MetaGraph cannot be parsed");

        std::map<std::string, const ::tensorflow::SignatureDef*> validSignatures = {};
        for (const auto& sit : m_metagraph_def->signature_def()) {
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

        m_graph_def = std::make_shared<::tensorflow::GraphDef>(m_metagraph_def->graph_def());

        // Update variables map using information by resolving AssignVariableOp graph nodes
        std::map<std::string, std::string> var_map;
        VariablesIndex::map_assignvariable(m_graph_def, var_map);
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
};  // GraphIteratorSavedModel

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
