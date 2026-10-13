// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/paged_qsa_indexer.hpp"

namespace ov::intel_gpu::op {

std::shared_ptr<Node> PagedQSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PagedQSAIndexer>(new_args, get_config());
}

}  // namespace ov::intel_gpu::op
