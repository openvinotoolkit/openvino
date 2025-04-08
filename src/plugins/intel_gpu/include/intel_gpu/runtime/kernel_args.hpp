// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory.hpp"

#include <vector>
#include <ostream>

namespace cldnn {

struct work_group_sizes {
    std::vector<size_t> global;
    std::vector<size_t> local;

    work_group_sizes() : global({1, 1, 1}), local({1, 1, 1}) {}
};

inline std::ostream& operator<<(std::ostream& os, const work_group_sizes& wgs) {
    os << "global: [" << wgs.global[0] << " " <<  wgs.global[1] << " " << wgs.global[2] << "] ";
    os << "local: [" << wgs.local[0] << " " <<  wgs.local[1] << " " << wgs.local[2] << "]";
    return os;
}

enum class kernel_language {
    OCLC,
    CM,
    OCLC_V2,
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scalar_desc {
    union ValueT {
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        int8_t s8;
        int16_t s16;
        int32_t s32;
        int64_t s64;
        float f32;
        double f64;
    };

    enum class Types {
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
    };

    Types t;
    ValueT v;
};

using scalars_desc = std::vector<scalar_desc>;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ArgumentDescpirtor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct argument_desc {
    enum class Types {
        INPUT,
        OUTPUT,
        WEIGHTS,
        BIAS,
        SCALE_TABLE,
        SLOPE,
        INTERNAL_BUFFER,
        SCALAR,
        CELL,       // LSTM cell input
        WEIGHTS_ZERO_POINTS,
        ACTIVATIONS_ZERO_POINTS,
        COMPENSATION,
        INPUT_OF_FUSED_PRIMITIVE,
        SHAPE_INFO
    };

    Types t;
    uint32_t index;
};

using arguments_desc = std::vector<argument_desc>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_arguments_desc {
    work_group_sizes workGroups;
    arguments_desc arguments;
    scalars_desc scalars;
    std::string layerID;
};

struct kernel_arguments_data {
    std::vector<memory::cptr> inputs;
    std::vector<memory::cptr> intermediates;
    std::vector<memory::cptr> outputs;
    memory::cptr weights;
    memory::cptr recurrent;
    memory::cptr hidden;
    memory::cptr cell;
    memory::cptr bias;
    memory::cptr weights_zero_points;
    memory::cptr activations_zero_points;
    memory::cptr compensation;
    memory::cptr lookup_table;
    memory::cptr scale_table;
    memory::cptr slope;
    memory::cptr shape_info;

    std::vector<memory::cptr> fused_op_inputs;
    const scalars_desc* scalars = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelString
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_string {
    std::string str;
    std::string jit;
    std::string undefs;
    std::string options;
    std::string entry_point;
    bool batch_compilation;
    bool has_microkernels;
    kernel_language language;

    kernel_string() : str(""), jit(""), undefs(""), options(""), entry_point(""),
    batch_compilation(false), has_microkernels(false), language(kernel_language::OCLC) {}

    std::string get_str() const { return str + jit + undefs + options + entry_point; }
    size_t get_hash() const { return std::hash<std::string>()(get_str()); }
};

}  // namespace cldnn
