// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/const_to_result_remover.hpp"

#include "helper_ops/unsupported_constant.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

bool ConstToResultRemover::run_on_model(const std::shared_ptr<ov::Model>& m) {
    // Note: need to perform this transformation only on the main ov::Model graph
    // no need to apply it for sub-graphs!
    ResultVector results_to_remove;
    // look for isolated UnsupportedConst->Result sub-graphs to remove
    // also, find isolated Constant->Result sub-graphs to remove
    for (const auto& result : m->get_results()) {
        auto unsupported_const = as_type_ptr<UnsupportedConstant>(result->get_input_node_shared_ptr(0));
        auto const_node = as_type_ptr<v0::Constant>(result->get_input_node_shared_ptr(0));
        if (unsupported_const || const_node) {
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
