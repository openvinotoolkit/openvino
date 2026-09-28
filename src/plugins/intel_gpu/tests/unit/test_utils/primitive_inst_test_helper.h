// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph/include/primitive_inst.h"

namespace cldnn {
// This class is intended to allow using private methods from primitive_inst within tests_core_internal project.
// Once needed, more methods wrapper should be added here.
class PrimitiveInstTestHelper {
public:
    static void set_allocation_done_by_other(const std::shared_ptr<primitive_inst>& inst, bool val) {
        inst->_allocation_done_by_other = val;
    }
    static void set_update_shape_done_by_other(const std::shared_ptr<primitive_inst>& inst, bool val) {
        inst->_update_shape_done_by_other = val;
    }
    static void do_runtime_in_place_crop(const std::shared_ptr<primitive_inst>& inst) {
        inst->do_runtime_in_place_crop();
    }
    static bool need_reset_output_memory(const std::shared_ptr<primitive_inst>& inst) {
        return inst->need_reset_output_memory();
    }
    static void update_weights(const std::shared_ptr<primitive_inst>& inst) {
        inst->update_weights();
    }
    static void cache_original_weights(const std::shared_ptr<primitive_inst>& inst) {
        const auto weights_idx = inst->get_node().get_primitive()->input.size();
        const auto original_weights = inst->dep_memory_ptr(weights_idx);
        inst->_reordered_weights_cache.add(original_weights->get_layout(), original_weights);
    }
};
}  // namespace cldnn
