// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "graph_iterator_proto.hpp"
#include "openvino/util/file_util.hpp"
#include "saved_model.pb.h"
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

    std::shared_ptr<VariablesIndex> get_variables_index() {
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
        std::basic_string<T> save_model_path = path + get_saved_model_name<T>();
        std::ifstream sm_stream{save_model_path.c_str(), std::ifstream::in | std::ifstream::binary};
        FRONT_END_GENERAL_CHECK(sm_stream && sm_stream.is_open(), "Model file does not exist");

        std::basic_string<T> varIndexPath = path + get_variables_index_name<T>();
        if (ov::util::file_exists(varIndexPath)) {
            m_variables_index = std::make_shared<VariablesIndex>();
            std::ifstream vi_stream{varIndexPath.c_str(), std::ifstream::in | std::ifstream::binary};
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

            // MetaGraph may have a list of signatures, but at this moment we need information only about
            // "serving_default" signature which contains information about inputs/outputs names for the
            // model. Situation when it is missing in a file also could be.
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
            VariablesIndex::map_assignvariable(m_graph_def, var_map);
            if (var_map.size() > 0 && m_variables_index.get() != nullptr) {
                for (auto var : var_map) {
                    m_variables_index->map_variable(var.first, var.second);
                }
            }

            initialize_decoders_and_library();

            return true;
        }

        FRONT_END_GENERAL_CHECK(false, "Saved Model doesn't contain MetaGraph with requested tag");

        return false;
    }
};  // GraphIteratorSavedModel

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
