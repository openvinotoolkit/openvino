// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <stdexcept>

#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/primitives/implementation_desc.hpp>

namespace bench_kernel {

// ============================================================================
// Data type helpers
// ============================================================================

inline cldnn::data_types str2dt(const std::string& s) {
    static const std::map<std::string, cldnn::data_types> dt_map = {
        {"f32",  cldnn::data_types::f32},
        {"f16",  cldnn::data_types::f16},
        {"i8",   cldnn::data_types::i8},
        {"u8",   cldnn::data_types::u8},
        {"i32",  cldnn::data_types::i32},
        {"i64",  cldnn::data_types::i64},
        {"i4",   cldnn::data_types::i4},
        {"u4",   cldnn::data_types::u4},
    };
    auto it = dt_map.find(s);
    if (it == dt_map.end()) {
        throw std::runtime_error("Unknown data type: " + s);
    }
    return it->second;
}

inline std::string dt2str(cldnn::data_types dt) {
    switch (dt) {
        case cldnn::data_types::f32:  return "f32";
        case cldnn::data_types::f16:  return "f16";
        case cldnn::data_types::i8:   return "i8";
        case cldnn::data_types::u8:   return "u8";
        case cldnn::data_types::i32:  return "i32";
        case cldnn::data_types::i64:  return "i64";
        case cldnn::data_types::i4:   return "i4";
        case cldnn::data_types::u4:   return "u4";
        default: return "unknown";
    }
}

// Parse colon-separated data types like "f16:i4:f16"
inline std::vector<cldnn::data_types> parse_dt_list(const std::string& s) {
    std::vector<cldnn::data_types> result;
    std::string token;
    for (char c : s) {
        if (c == ':') {
            if (!token.empty()) {
                result.push_back(str2dt(token));
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        result.push_back(str2dt(token));
    }
    return result;
}

// ============================================================================
// Impl types (using OV GPU plugin's cldnn::impl_types)
// ============================================================================

using impl_type = cldnn::impl_types;

inline impl_type str2impl(const std::string& s) {
    if (s == "any" || s.empty())   return impl_type::any;
    if (s == "ocl")                return impl_type::ocl;
    if (s == "onednn")             return impl_type::onednn;
    if (s == "cpu")                return impl_type::cpu;
    if (s == "common")             return impl_type::common;
    throw std::runtime_error("Unknown impl type: " + s);
}

inline std::string impl2str(impl_type t) {
    switch (t) {
        case impl_type::any:    return "any";
        case impl_type::ocl:    return "ocl";
        case impl_type::onednn: return "onednn";
        case impl_type::cpu:    return "cpu";
        case impl_type::common: return "common";
        default: return "unknown";
    }
}

// ============================================================================
// Shape parsing helpers
// ============================================================================

// Parse a single dimension string like "1x4096" into vector<int64_t>
inline std::vector<int64_t> parse_dims(const std::string& s) {
    std::vector<int64_t> dims;
    std::string token;
    for (char c : s) {
        if (c == 'x' || c == 'X') {
            if (!token.empty()) {
                dims.push_back(std::stoll(token));
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        dims.push_back(std::stoll(token));
    }
    return dims;
}

// Parse colon-separated tensor shapes like "1x4096:4096x4096"
// Empty tokens between colons are treated as scalar shape {1}
// (e.g., ":" → [{1},{1}], ":4x4" → [{1},{4,4}], "4x4:" → [{4,4},{1}])
inline std::vector<std::vector<int64_t>> parse_shapes(const std::string& s) {
    std::vector<std::vector<int64_t>> result;
    if (s.empty()) return result;

    std::string token;
    for (char c : s) {
        if (c == ':') {
            if (token.empty()) {
                result.push_back({1});  // 0-dim scalar → treat as shape {1}
            } else {
                result.push_back(parse_dims(token));
                token.clear();
            }
        } else {
            token += c;
        }
    }
    // Handle last token (or trailing colon)
    if (token.empty()) {
        // Trailing colon means last shape was empty (scalar)
        // Only add if we already have at least one shape (i.e., there was a colon)
        if (!result.empty()) {
            result.push_back({1});
        }
    } else {
        result.push_back(parse_dims(token));
    }
    return result;
}

// ============================================================================
// String split utility
// ============================================================================

inline std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::string token;
    for (char c : s) {
        if (c == delim) {
            if (!token.empty()) result.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) result.push_back(token);
    return result;
}

}  // namespace bench_kernel
