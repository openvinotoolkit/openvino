// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <initializer_list>
#include <vector>
#include <string>
#include "llm_types.hpp"

namespace llmdnn {

inline size_t get_precision_size(data_type_t type) {
    switch(type) {
        case dnnl_f16:
        case dnnl_bf16:
            return 2;
        case dnnl_f32:
        case dnnl_s32:
            return 4;
        case dnnl_s8:
        case dnnl_u8:
            return 1;
        case dnnl_f64:
            return 8;
        default:
            assert(false && "unknown data type");
            return 0;
    }
}

inline data_type_t get_dt_from_str(const std::string& name) {
    static std::pair<const char*, data_type_t> name2type[] = {
        { "f16", dnnl_f16 },
        { "bf16", dnnl_bf16 },
        { "f32", dnnl_f32 },
        { "s32", dnnl_s32 },
        { "i32", dnnl_s32 },
        { "s8", dnnl_s8 },
        { "i8", dnnl_s8 },
        { "u8", dnnl_u8 },
        { "f64", dnnl_f64 },
    };
    for (size_t i = 0; i < sizeof(name2type) / sizeof(name2type[0]); i++) {
        if (name == name2type[i].first)
            return name2type[i].second;
    }

    return dnnl_data_type_undef;
}

}