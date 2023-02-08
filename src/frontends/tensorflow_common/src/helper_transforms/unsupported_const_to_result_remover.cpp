// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/unsupported_const_to_result_remover.hpp"

#include "helper_ops/unsupported_constant.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

bool UnsupportedConstToResultRemover::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ResultVector results_to_remove;
    // look for isolated UnsupportedConst->Result sub-graphs to remove
    for (const auto& result : m->get_results()) {
        auto unsupported_const = as_type_ptr<UnsupportedConstant>(result->get_input_node_shared_ptr(0));
        if (unsupported_const && unsupported_const->output(0).get_target_inputs().size() == 1) {
            results_to_remove.push_back(result);
        }
    }

    for (const auto& result : results_to_remove) {
        m->remove_result(result);
    }

    return true;
}

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
