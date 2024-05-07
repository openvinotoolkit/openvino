// Copyright (C) 2024 Intel Corporation
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
    HashTableKeysValuesMap m_hash_table_keys_map;
    HashTableKeysValuesMap m_hash_table_values_map;
    bool m_mmap_enabled;

public:
    template <typename T>
    GraphIteratorMeta(const std::basic_string<T>& path, const bool mmap_enabled)
        : m_metagraph_def(std::make_shared<::tensorflow::MetaGraphDef>()),
          m_mmap_enabled(mmap_enabled) {
        this->read_meta(path);
    }

    template <typename T>
    static bool is_supported(const std::basic_string<T>& path) {
        try {
            std::ifstream mg_stream(path.c_str(), std::ios::in | std::ifstream::binary);
            auto metagraph_def = std::make_shared<::tensorflow::MetaGraphDef>();
            return mg_stream && mg_stream.is_open() && metagraph_def->ParsePartialFromIstream(&mg_stream) &&
                   metagraph_def->has_graph_def() && metagraph_def->graph_def().node_size() > 0;
        } catch (...) {
            return false;
        }
    }

    std::shared_ptr<VariablesIndex> get_variables_index() {
        return m_variables_index;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_metagraph_input_names() const {
        return m_inputs_map;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_metagraph_output_names() const {
        return m_outputs_map;
    }

    HashTableKeysValuesMap get_hash_table_keys_map() const {
        return m_hash_table_keys_map;
    }

    HashTableKeysValuesMap get_hash_table_values_map() const {
        return m_hash_table_values_map;
    }

private:
    bool is_valid_signature(const ::tensorflow::SignatureDef& signature) const;

    template <typename T>
    bool read_meta(const std::basic_string<T>& path) {
        std::basic_string<T> model_path = path.substr(0, path.find_last_of('.'));

        std::ifstream mg_stream{path.c_str(), std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(mg_stream && mg_stream.is_open(), "Model file does not exist");

        std::basic_string<T> varIndexPath = get_variables_index_name<T>(model_path);
        if (ov::util::file_exists(varIndexPath)) {
            m_variables_index = std::make_shared<VariablesIndex>(m_mmap_enabled);
            std::ifstream vi_stream{varIndexPath.c_str(), std::ifstream::in | std::ifstream::binary};
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
            /*
                "serving_default" signature contains map of input/output names.
                Here we are storing two maps for inputs and outputs.
                Map looks like "name_set_by_user" = "internal_name:port".
                For example, "input_mask" = "serving_default_input_mask:0"
            */
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
        VariablesIndex::map_assignvariable(m_graph_def, var_map, m_hash_table_keys_map, m_hash_table_values_map);
        for (auto var : var_map) {
            m_variables_index->map_variable(var.first, var.second);
        }

        initialize_decoders_and_library();

        return true;
    }
};  // GraphIteratorMeta

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
