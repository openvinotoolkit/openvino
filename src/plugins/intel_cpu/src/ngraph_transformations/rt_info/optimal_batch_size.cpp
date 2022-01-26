// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include "optimal_batch_size.hpp"

namespace MKLDNNPlugin {
bool has_optimal_bs(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().count(OptimalBatchSize::get_type_info_static());
}
size_t get_optimal_bs(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().at(OptimalBatchSize::get_type_info_static()).as<OptimalBatchSize>().get_value();
}
void set_optimal_bs(const std::shared_ptr<ov::Node>& node, const size_t opt_batch) {
    node->get_rt_info().emplace(OptimalBatchSize::get_type_info_static(), OptimalBatchSize{opt_batch});
}
}  // namespace MKLDNNPlugin