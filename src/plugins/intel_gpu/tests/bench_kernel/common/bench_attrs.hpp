// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include "bench_types.hpp"

namespace bench_kernel {

// ============================================================================
// Scale/ZeroPoint attribute entry
//   --attr-scales=ARG:POLICY[:DATA_TYPE[:GROUPS]]
//   --attr-zero-points=ARG:POLICY[:DATA_TYPE[:GROUPS]]
// ============================================================================

enum class attr_arg {
    src,    // activation / input
    wei,    // weights
    dst,    // output
    key,    // SDPA key
    value,  // SDPA value
};

inline attr_arg str2arg(const std::string& s) {
    if (s == "src" || s == "src0") return attr_arg::src;
    if (s == "wei" || s == "weights") return attr_arg::wei;
    if (s == "dst") return attr_arg::dst;
    if (s == "key") return attr_arg::key;
    if (s == "value") return attr_arg::value;
    throw std::runtime_error("Unknown attr arg: " + s);
}

inline std::string arg2str(attr_arg a) {
    switch (a) {
        case attr_arg::src:   return "src";
        case attr_arg::wei:   return "wei";
        case attr_arg::dst:   return "dst";
        case attr_arg::key:   return "key";
        case attr_arg::value: return "value";
    }
    return "unknown";
}

enum class attr_policy {
    common,       // per-tensor (single value)
    per_oc,       // per output channel
    per_ic,       // per input channel
    per_ocic,     // per output+input channel (grouped)
    per_token,    // per token (used in dynamic quantization)
    per_tensor,   // alias for common in some contexts
};

inline attr_policy str2policy(const std::string& s) {
    if (s == "common" || s == "per_tensor") return attr_policy::common;
    if (s == "per_oc")    return attr_policy::per_oc;
    if (s == "per_ic")    return attr_policy::per_ic;
    if (s == "per_ocic")  return attr_policy::per_ocic;
    if (s == "per_token") return attr_policy::per_token;
    throw std::runtime_error("Unknown attr policy: " + s);
}

inline std::string policy2str(attr_policy p) {
    switch (p) {
        case attr_policy::common:    return "common";
        case attr_policy::per_oc:    return "per_oc";
        case attr_policy::per_ic:    return "per_ic";
        case attr_policy::per_ocic:  return "per_ocic";
        case attr_policy::per_token: return "per_token";
        case attr_policy::per_tensor:return "per_tensor";
    }
    return "unknown";
}

struct scale_entry {
    attr_arg      arg;
    attr_policy   policy;
    cldnn::data_types dt = cldnn::data_types::f16;
    std::vector<int64_t> groups;  // e.g., {1, 128} for grouped quantization
};

struct zero_point_entry {
    attr_arg      arg;
    attr_policy   policy;
    cldnn::data_types dt = cldnn::data_types::i8;
    std::vector<int64_t> groups;
};

// Parse --attr-scales string into vector of scale_entry
// Format: ARG:POLICY[:DATA_TYPE[:GROUPS]][+ARG:POLICY[:DATA_TYPE[:GROUPS]]]
inline std::vector<scale_entry> parse_scales(const std::string& s) {
    std::vector<scale_entry> result;
    if (s.empty()) return result;

    auto entries = split(s, '+');
    for (const auto& entry_str : entries) {
        auto parts = split(entry_str, ':');
        if (parts.size() < 2) {
            throw std::runtime_error("Invalid scale entry: " + entry_str);
        }
        scale_entry e;
        e.arg    = str2arg(parts[0]);
        e.policy = str2policy(parts[1]);

        if (parts.size() >= 3) {
            e.dt = str2dt(parts[2]);
        }
        if (parts.size() >= 4) {
            e.groups = parse_dims(parts[3]);
        }
        result.push_back(e);
    }
    return result;
}

// Parse --attr-zero-points string
inline std::vector<zero_point_entry> parse_zero_points(const std::string& s) {
    std::vector<zero_point_entry> result;
    if (s.empty()) return result;

    auto entries = split(s, '+');
    for (const auto& entry_str : entries) {
        auto parts = split(entry_str, ':');
        if (parts.size() < 2) {
            throw std::runtime_error("Invalid zero_point entry: " + entry_str);
        }
        zero_point_entry e;
        e.arg    = str2arg(parts[0]);
        e.policy = str2policy(parts[1]);

        if (parts.size() >= 3) {
            e.dt = str2dt(parts[2]);
        }
        if (parts.size() >= 4) {
            e.groups = parse_dims(parts[3]);
        }
        result.push_back(e);
    }
    return result;
}

// ============================================================================
// Post-ops attribute
//   --attr-post-ops=POST_OP[+POST_OP[+...]]
//
//   Activation:  FUNC[:ALPHA[:BETA]]
//   Eltwise:     sum|prod|sub|div:DATA_TYPE[:BROADCAST]
//   Quantize:    quantize:OUT_DT[:SCALE_TYPE[:ZP_TYPE]]
//   SwiGLU:      swiglu:SPLIT_LEN:SPLIT_IDX
// ============================================================================

enum class post_op_kind {
    activation,
    eltwise,
    quantize,
    swiglu,
};

// Activation function names
enum class activation_func {
    relu, sigmoid, tanh, elu, abs, sqrt, square, exp, log,
    gelu_erf, gelu_tanh, swish, mish, hardswish, hardsigmoid,
    softplus, clip, round, linear, clamp,
};

inline activation_func str2activation(const std::string& s) {
    static const std::map<std::string, activation_func> map = {
        {"relu", activation_func::relu},
        {"sigmoid", activation_func::sigmoid},
        {"tanh", activation_func::tanh},
        {"elu", activation_func::elu},
        {"abs", activation_func::abs},
        {"sqrt", activation_func::sqrt},
        {"square", activation_func::square},
        {"exp", activation_func::exp},
        {"log", activation_func::log},
        {"gelu_erf", activation_func::gelu_erf},
        {"gelu_tanh", activation_func::gelu_tanh},
        {"swish", activation_func::swish},
        {"mish", activation_func::mish},
        {"hardswish", activation_func::hardswish},
        {"hardsigmoid", activation_func::hardsigmoid},
        {"softplus", activation_func::softplus},
        {"clip", activation_func::clip},
        {"clamp", activation_func::clamp},
        {"round", activation_func::round},
        {"linear", activation_func::linear},
    };
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    throw std::runtime_error("Unknown activation: " + s);
}

inline bool is_activation_name(const std::string& s) {
    static const std::vector<std::string> names = {
        "relu", "sigmoid", "tanh", "elu", "abs", "sqrt", "square", "exp", "log",
        "gelu_erf", "gelu_tanh", "swish", "mish", "hardswish", "hardsigmoid",
        "softplus", "clip", "clamp", "round", "linear",
    };
    for (const auto& n : names) {
        if (s == n) return true;
    }
    return false;
}

// Eltwise mode for post-op
enum class eltwise_mode {
    sum,
    prod,
    sub,
    div_mode,
};

inline eltwise_mode str2eltwise_mode(const std::string& s) {
    if (s == "sum")  return eltwise_mode::sum;
    if (s == "prod") return eltwise_mode::prod;
    if (s == "sub")  return eltwise_mode::sub;
    if (s == "div")  return eltwise_mode::div_mode;
    throw std::runtime_error("Unknown eltwise mode: " + s);
}

inline bool is_eltwise_mode(const std::string& s) {
    return s == "sum" || s == "prod" || s == "sub" || s == "div";
}

enum class broadcast_spec {
    none,       // same shape
    per_oc,     // per output channel
    per_tensor, // scalar
};

inline broadcast_spec str2broadcast(const std::string& s) {
    if (s == "per_oc")     return broadcast_spec::per_oc;
    if (s == "per_tensor") return broadcast_spec::per_tensor;
    return broadcast_spec::none;
}

struct post_op_entry {
    post_op_kind kind;
    bool skip = false;  // true for non-computational fused ops (e.g., reorder)

    // Activation params
    activation_func act_func = activation_func::relu;
    float alpha = 0.0f;
    float beta  = 0.0f;

    // Eltwise params
    eltwise_mode elt_mode = eltwise_mode::sum;
    cldnn::data_types elt_dt = cldnn::data_types::f16;
    broadcast_spec elt_broadcast = broadcast_spec::none;

    // Quantize params
    cldnn::data_types quant_out_dt = cldnn::data_types::u8;
    bool quant_dt_explicit = false;  // true if quant_out_dt was explicitly specified
    std::string quant_scale_type;
    std::string quant_zp_type;

    // SwiGLU params
    int swiglu_split_length = 2;
    int swiglu_split_to_glu_idx = 0;
};

// Parse a single post-op token
inline post_op_entry parse_single_post_op(const std::string& token) {
    auto parts = split(token, ':');
    if (parts.empty()) {
        throw std::runtime_error("Empty post-op token");
    }

    post_op_entry e;

    // Check if it's an activation
    if (is_activation_name(parts[0])) {
        e.kind = post_op_kind::activation;
        e.act_func = str2activation(parts[0]);
        if (parts.size() >= 2) e.alpha = std::stof(parts[1]);
        if (parts.size() >= 3) e.beta  = std::stof(parts[2]);
        return e;
    }

    // Check if it's an eltwise
    if (is_eltwise_mode(parts[0])) {
        e.kind = post_op_kind::eltwise;
        e.elt_mode = str2eltwise_mode(parts[0]);
        if (parts.size() >= 2) e.elt_dt = str2dt(parts[1]);
        if (parts.size() >= 3) e.elt_broadcast = str2broadcast(parts[2]);
        return e;
    }

    // Quantize
    if (parts[0] == "quantize") {
        e.kind = post_op_kind::quantize;
        if (parts.size() >= 2) { e.quant_out_dt = str2dt(parts[1]); e.quant_dt_explicit = true; }
        if (parts.size() >= 3) e.quant_scale_type = parts[2];
        if (parts.size() >= 4) e.quant_zp_type = parts[3];
        return e;
    }

    // SwiGLU
    if (parts[0] == "swiglu") {
        e.kind = post_op_kind::swiglu;
        if (parts.size() >= 2) e.swiglu_split_length = std::stoi(parts[1]);
        if (parts.size() >= 3) e.swiglu_split_to_glu_idx = std::stoi(parts[2]);
        return e;
    }

    // "reorder" is a type-conversion fused op, not a computational post-op → skip
    if (parts[0] == "reorder") {
        e.kind = post_op_kind::activation;  // dummy, will be filtered by empty check
        e.skip = true;
        return e;
    }

    throw std::runtime_error("Unknown post-op: " + parts[0]);
}

// Parse --attr-post-ops string
// Format: POST_OP[+POST_OP[+...]]
inline std::vector<post_op_entry> parse_post_ops(const std::string& s) {
    std::vector<post_op_entry> result;
    if (s.empty()) return result;

    // Split on '+' but respect scientific notation (e.g., -6.78e+07).
    // Scientific notation pattern: DIGIT followed by e/E followed by +/-.
    // We must NOT treat the '+' in "quantize+" as scientific notation just
    // because 'quantize' ends with 'e'.
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+' && cur.size() >= 2) {
            // Check if this '+' is part of scientific notation: ...DIGITe+ or ...DIGITE+
            char prev_e = cur[cur.size() - 1];
            char prev_d = cur[cur.size() - 2];
            if ((prev_e == 'e' || prev_e == 'E') && (prev_d >= '0' && prev_d <= '9')) {
                cur += s[i];  // keep the '+' as part of the number
                continue;
            }
        }
        if (s[i] == '+') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        } else {
            cur += s[i];
        }
    }
    if (!cur.empty()) tokens.push_back(cur);

    for (const auto& token : tokens) {
        auto entry = parse_single_post_op(token);
        if (!entry.skip) {
            result.push_back(entry);
        }
    }
    return result;
}

// Resolve quantize output dtype for post-ops that don't have an explicit type.
// When quantize appears without ":i8" or ":u8", infer the target integer type from
// the primitive's output dtype or input dtype. This matches GPU behavior where quantize
// fused ops convert to the actual output integer type, not always u8.
inline void resolve_quantize_dtypes(std::vector<post_op_entry>& post_ops,
                                     cldnn::data_types out_dt,
                                     cldnn::data_types src_dt) {
    auto is_integer_dt = [](cldnn::data_types dt) {
        return dt == cldnn::data_types::i8  || dt == cldnn::data_types::u8 ||
               dt == cldnn::data_types::i4  || dt == cldnn::data_types::u4 ||
               dt == cldnn::data_types::i32 || dt == cldnn::data_types::i64;
    };

    // Determine the quantize target type: prefer output dtype, then input dtype
    cldnn::data_types inferred_dt = cldnn::data_types::u8;
    if (is_integer_dt(out_dt)) {
        inferred_dt = out_dt;
    } else if (is_integer_dt(src_dt)) {
        inferred_dt = src_dt;
    }

    for (auto& po : post_ops) {
        if (po.kind == post_op_kind::quantize && !po.quant_dt_explicit) {
            po.quant_out_dt = inferred_dt;
        }
    }
}

}  // namespace bench_kernel
