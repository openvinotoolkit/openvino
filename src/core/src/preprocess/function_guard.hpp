// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace preprocess {

/// \brief Internal guard to make preprocess builder exception-safe
class FunctionGuard {
    std::shared_ptr<Model> m_function;
    ParameterVector m_parameters;
    ResultVector m_results;
    std::vector<std::unordered_set<std::string>> m_result_tensors;
    std::map<std::shared_ptr<op::v0::Parameter>, std::set<Input<Node>>> m_backup;
    bool m_done = false;

public:
    FunctionGuard(const std::shared_ptr<Model>& f) : m_function(f) {
        m_parameters = f->get_parameters();
        for (const auto& param : f->get_parameters()) {
            m_backup.insert({param, param->output(0).get_target_inputs()});
        }
        m_results = f->get_results();
        for (const auto& result : m_results) {
            m_result_tensors.push_back(result->get_default_output().get_tensor().get_names());
        }
    }
    virtual ~FunctionGuard() {
        if (!m_done) {
            try {
                auto params = m_function->get_parameters();
                // Remove parameters added by preprocessing
                for (const auto& param : params) {
                    m_function->remove_parameter(param);
                }
                // Insert old parameters and update consumers
                for (const auto& item : m_backup) {
                    // Replace consumers
                    for (auto consumer : item.second) {
                        consumer.replace_source_output(item.first);
                    }
                }
                m_function->add_parameters(m_parameters);

                auto results = m_function->get_results();

                // Remove results added by postprocessing
                for (const auto& result : results) {
                    m_function->remove_result(result);
                }
                // Restore removed tensor names
                for (size_t i = 0; i < m_results.size(); ++i) {
                    m_results[i]->get_default_output().get_tensor().set_names(m_result_tensors[i]);
                }
                m_function->add_results(m_results);
            } catch (std::exception& ex) {
                // Stress condition, can't recover function to original state
                std::cerr << "Unrecoverable error occurred during preprocessing. Model is corrupted, exiting: "
                          << ex.what();
                // exit(EXIT_FAILURE);
            }
        }
    }
    void reset() noexcept {
        m_done = true;
    }
};

}  // namespace preprocess
}  // namespace ov
