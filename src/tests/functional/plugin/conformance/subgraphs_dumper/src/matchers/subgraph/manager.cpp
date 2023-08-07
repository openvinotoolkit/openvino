// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include "matchers/subgraph/manager.hpp"

using namespace ov::tools::subgraph_dumper;

bool ExtractorsManager::match(const std::shared_ptr<ov::Model> &model,
                              const std::shared_ptr<ov::Model> &ref) {
    for (const auto &it : m_extractors) {
        if (it.second->match(model, ref)) {
            return true;
        }
    }
    return false;
}

bool ExtractorsManager::match(const std::shared_ptr<ov::Model> &model,
                              const std::shared_ptr<ov::Model> &ref,
                              std::map<std::string, InputInfo> &in_info,
                              const std::map<std::string, InputInfo> &in_info_ref) {
    if (match(model, ref)) {
        try {
            auto new_input_info = align_input_info(model, ref, in_info, in_info_ref);
            in_info = new_input_info;
            return true;
        } catch (std::exception& e) {
            return false;
        }
    }
    return false;
}

std::map<std::string, InputInfo>
ExtractorsManager::align_input_info(const std::shared_ptr<ov::Model>& model,
                                    const std::shared_ptr<ov::Model>& model_ref,
                                    const std::map<std::string, InputInfo>& in_info,
                                    const std::map<std::string, InputInfo>& in_info_ref) {
    std::map<std::string, InputInfo> new_input_info = in_info;
    bool is_update_required = false;
    for (const auto& in_info_item : in_info_ref) {
        if (!in_info.count(in_info_item.first)) {
            is_update_required = true;
            break;
        }
    }
    if (is_update_required) {
        std::map<std::string, InputInfo> new_ref_input_info = in_info_ref;
        // align matched model names
        auto ref_model_ops = model_ref->get_ordered_ops();
        auto model_ops = model->get_ordered_ops();
        size_t ordered_ops_size = model_ops.size();
        if (ordered_ops_size != ref_model_ops.size()) {
            throw std::runtime_error("Matched models are different!");
        }
        for (size_t i = 0; i < ordered_ops_size; ++i) {
            auto model_op_name = model_ops[i]->get_friendly_name();
            auto model_ref_op_name = ref_model_ops[i]->get_friendly_name();
            if (in_info.count(model_op_name)) {
                auto input_info = new_input_info[model_op_name];
                if (input_info.is_const != new_ref_input_info[model_ref_op_name].is_const) {
                    throw std::runtime_error("Impossible yo update input info!!!");
                }
                new_input_info.erase(model_op_name);
                new_input_info.insert({ model_ref_op_name, input_info });
            }
        }
    }
    return new_input_info;
}

std::list<ExtractedPattern>
ExtractorsManager::extract(const std::shared_ptr<ov::Model> &model, bool is_extract_body) {
    std::list<ExtractedPattern> result;
    for (const auto &it : m_extractors) {
        auto start = std::chrono::high_resolution_clock::now();
        it.second->set_extractor_name(it.first);
        auto extracted_patterns = it.second->extract(model, is_extract_body);
        result.insert(result.end(), extracted_patterns.begin(), extracted_patterns.end());
        auto end = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[ INFO ][ EXTRACTOR DURATION ] " << it.first << " " << delta << "ms" << std::endl;
    }
    return result;
}
