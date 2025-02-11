// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <impls/onednn/utils.hpp>
#endif

using namespace cldnn;

void add_onednn_optimization_attributes::run(program& p) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    for (auto& node : p.get_processing_order()) {
        if (node->get_preferred_impl_type() == impl_types::onednn
            && !node->is_dynamic()) {
            node->init_onednn_primitive_attributes();
        }
    }
#endif // ENABLE_ONEDNN_FOR_GPU
}
