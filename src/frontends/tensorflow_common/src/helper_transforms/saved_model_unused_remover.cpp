// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/saved_model_unused_remover.hpp"

#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

bool SavedModelUnusedRemover::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ParameterVector params_to_remove;
    ResultVector results_to_remove;

    // There is two cases
    // 1. When we found unused result with/without unused parameter
    // 2. When we found unused parameter

    for (const auto& result : m->get_results()) {
        bool isUsed = false;
        for (size_t i = 0; i < result->get_input_size(); ++i) {
            const auto& node_names = result->get_input_tensor(i).get_names();
            isUsed |= std::find(node_names.begin(), node_names.end(), "saved_model_unused") == node_names.end();
        }
        if (!isUsed) {
            results_to_remove.push_back(result);
            continue;
        }

        auto param = as_type_ptr<v0::Parameter>(result->get_input_node_shared_ptr(0));
        if (param) {
            isUsed = false;
            for (size_t i = 0; i < param->get_output_size(); ++i) {
                const auto& node_names = param->get_output_tensor(i).get_names();
                isUsed |= std::find(node_names.begin(), node_names.end(), "saved_model_unused") == node_names.end();
            }
            if (!isUsed) {
                results_to_remove.push_back(result);
                params_to_remove.push_back(param);
            }
        }
    }

    for (const auto& param : m->get_parameters()) {
        bool isUsed = false;
        for (size_t i = 0; i < param->get_output_size(); ++i) {
            const auto& node_names = param->get_output_tensor(i).get_names();
            isUsed |= std::find(node_names.begin(), node_names.end(), "saved_model_unused") == node_names.end();
        }
        if (!isUsed && std::find(params_to_remove.begin(), params_to_remove.end(), param) == params_to_remove.end()) {
            params_to_remove.push_back(param);
        }
    }

    for (const auto& result : results_to_remove) {
        m->remove_result(result);
    }

    for (const auto& param : params_to_remove) {
        m->remove_parameter(param);
    }

    return true;
}

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
