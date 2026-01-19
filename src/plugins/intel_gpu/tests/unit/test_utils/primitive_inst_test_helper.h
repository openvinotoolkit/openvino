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
    static bool need_reset_output_memory(const std::shared_ptr<primitive_inst>& inst) {
        return inst->need_reset_output_memory();
    }
};
}  // namespace cldnn
