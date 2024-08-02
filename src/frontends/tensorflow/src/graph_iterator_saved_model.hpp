// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "graph_iterator_proto.hpp"
#include "openvino/util/file_util.hpp"
#include "ov_tensorflow/saved_model.pb.h"
#include "variables_index.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

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

// Loads graph from Tensorflow Saved Model file (saved_model.pb)
class GraphIteratorSavedModel : public GraphIteratorProto {
    std::shared_ptr<::tensorflow::SavedModel> m_saved_model;
    std::shared_ptr<VariablesIndex> m_variables_index;
    std::shared_ptr<std::map<std::string, std::string>> m_inputs_map;
    std::shared_ptr<std::map<std::string, std::string>> m_outputs_map;
    HashTableKeysValuesMap m_hash_table_keys_map;
    HashTableKeysValuesMap m_hash_table_values_map;
    bool m_mmap_enabled;

public:
    template <typename T>
    GraphIteratorSavedModel(const std::basic_string<T>& path, const std::string& tags, const bool mmap_enabled)
        : m_saved_model(std::make_shared<::tensorflow::SavedModel>()),
          m_mmap_enabled(mmap_enabled) {
        this->read_saved_model(path, tags);
    }

    static bool is_supported(const std::string& path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    static bool is_supported(const std::wstring& path);
#endif

    std::shared_ptr<VariablesIndex> get_variables_index() {
        return m_variables_index;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_saved_model_input_names() const {
        return m_inputs_map;
    }

    std::shared_ptr<std::map<std::string, std::string>> get_saved_model_output_names() const {
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
    bool read_saved_model(const std::basic_string<T>& path, const std::string& tags) {
        std::basic_string<T> save_model_path = path + get_saved_model_name<T>();
        std::ifstream sm_stream{save_model_path.c_str(), std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(sm_stream && sm_stream.is_open(), "[TensorFlow Frontend] Model file does not exist");

        std::basic_string<T> varIndexPath = path + get_variables_index_name<T>();
        if (ov::util::file_exists(varIndexPath)) {
            m_variables_index = std::make_shared<VariablesIndex>(m_mmap_enabled);
            std::ifstream vi_stream{varIndexPath.c_str(), std::ifstream::in | std::ifstream::binary};
            FRONT_END_GENERAL_CHECK(vi_stream && vi_stream.is_open(),
                                    "[TensorFlow Frontend] Saved Model's variable index file does not exist");
            FRONT_END_GENERAL_CHECK(m_variables_index->read_variables(vi_stream, path),
                                    "[TensorFlow Frontend] Saved Model's variable index file cannot be parsed");
        }

        bool res = m_saved_model->ParseFromIstream(&sm_stream);
        FRONT_END_GENERAL_CHECK(res && m_saved_model->meta_graphs_size(),
                                "[TensorFlow Frontend] Saved Model cannot be parsed");

        auto tag_list = split_tags(tags);

        // SavedModel can contain several MetaGraph with different tags. Look for MetaGraph with the required tag
        for (const auto& meta_graph : m_saved_model->meta_graphs()) {
            if (!meta_graph.has_graph_def()) {
                continue;
            }

            bool tag_found = false;

            if (meta_graph.meta_info_def().tags_size() > 0) {
                tag_found = std::all_of(meta_graph.meta_info_def().tags().begin(),
                                        meta_graph.meta_info_def().tags().end(),
                                        [&tag_list](const std::string& tag) {
                                            return std::find(tag_list.begin(), tag_list.end(), tag) != tag_list.end();
                                        });
            }

            if (tag_found) {
                return load_meta_graph(meta_graph);
            }
        }

        // Alternate behavior for working with "default tag" to support additional cases for read_model
        if (tags == META_GRAPH_DEFAULT_TAG) {
            // If we have only one MetaGraph - try to use it
            if (m_saved_model->meta_graphs_size() == 1 && m_saved_model->meta_graphs(0).has_graph_def()) {
                return load_meta_graph(m_saved_model->meta_graphs(0));
            }

            // If MetaGraph with tag == META_GRAPH_DEFAULT_TAG already found - we shouldn't reach this place.
            // Otherwise we try to find a MetaGraph with no tags as an alternative
            for (const auto& meta_graph : m_saved_model->meta_graphs()) {
                if (!meta_graph.has_graph_def()) {
                    continue;
                }

                if (meta_graph.meta_info_def().tags_size() == 0) {
                    return load_meta_graph(meta_graph);
                }
            }

            FRONT_END_GENERAL_CHECK(false,
                                    "[TensorFlow Frontend] Saved Model doesn't contain any applicable MetaGraph");
        }

        FRONT_END_GENERAL_CHECK(false,
                                "[TensorFlow Frontend] Saved Model doesn't contain MetaGraph with requested tag");

        return false;
    }

    /// \brief Does a loading of exact meta-graph
    bool load_meta_graph(const ::tensorflow::MetaGraphDef& meta_graph) {
        std::map<std::string, const ::tensorflow::SignatureDef*> validSignatures = {};
        for (const auto& sit : meta_graph.signature_def()) {
            const std::string& key = sit.first;
            const ::tensorflow::SignatureDef& val = sit.second;
            if (is_valid_signature(val)) {
                validSignatures[key] = &val;
            }
        }

        // MetaGraph may have a list of signatures, but at this moment we need information only about
        // "serving_default" signature which contains information about inputs/outputs names for the
        // model. Situation when it is missing in a file also could be.
        auto serving_default = validSignatures.find("serving_default");

        if (serving_default != validSignatures.end()) {
            // "serving_default" signature contains map of input/output names.
            // here we are storing two maps for inputs and outputs.
            // map looks like {"internal_name:port": "name_set_by_user"}
            // for example, {"serving_default_input_mask:0": "input_mask"}
            m_inputs_map = std::make_shared<std::map<std::string, std::string>>();
            m_outputs_map = std::make_shared<std::map<std::string, std::string>>();
            for (const auto& input : serving_default->second->inputs()) {
                (*m_inputs_map)[input.second.name()] = input.first;
            }
            for (const auto& output : serving_default->second->outputs()) {
                (*m_outputs_map)[output.second.name()] = output.first;
            }
        } else if (validSignatures.size() > 0) {
            // no special signature for serving
            // so use all inputs and outputs for all signatures
            m_inputs_map = std::make_shared<std::map<std::string, std::string>>();
            m_outputs_map = std::make_shared<std::map<std::string, std::string>>();
            for (const auto& signature : validSignatures) {
                for (const auto& input : signature.second->inputs()) {
                    (*m_inputs_map)[input.second.name()] = input.first;
                }
                for (const auto& output : signature.second->outputs()) {
                    (*m_outputs_map)[output.second.name()] = output.first;
                }
            }
        }

        m_graph_def = std::make_shared<::tensorflow::GraphDef>(meta_graph.graph_def());

        // Update variables map using information by resolving AssignVariableOp graph nodes
        std::map<std::string, std::string> var_map;
        VariablesIndex::map_assignvariable(m_graph_def, var_map, m_hash_table_keys_map, m_hash_table_values_map);
        if (var_map.size() > 0 && m_variables_index.get() != nullptr) {
            for (auto var : var_map) {
                m_variables_index->map_variable(var.first, var.second);
            }
        }

        initialize_decoders_and_library();

        return true;
    }

    /// \brief Splitting tags by using "," delimeter
    /// \param[in] tags String with tags separated by ","
    /// \return Returns vector with splitted tags, no trimming is used. When you pass "tag1, tag2"
    /// you will have a vector ["tag1", " tag2"]. Because TensorFlow saves tags without trimming
    std::vector<std::string> split_tags(const std::string tags) const;
};  // GraphIteratorSavedModel

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
