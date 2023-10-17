// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_snippets_emitters.hpp"

namespace ov {
namespace intel_cpu {

#define GET_OFF_DYN(field) offsetof(jit_snippets_dynamic_call_args, field)
class jit_snippets_call_args_dynamic : public jit_snippets_call_args {
public:
    class loop_args {
        int32_t work_amount = 0;
        int32_t num_data_ptrs = 0;
        int32_t* ptr_increments = nullptr;
        int32_t* finalization_offsets = nullptr;
    };
    int32_t num_loops = 0;
    loop_args* loop_args = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
