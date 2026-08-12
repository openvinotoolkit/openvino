// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace ov {
namespace npuw {
namespace perf {
namespace tags {

// Profiling tags are '/'-separated paths: the map ordering is then already the tree
// order and report() can print the hierarchy without storing parent links.
// All of these are built at most once per (submodel, level) per request and cached
// as metric handles - never on the measured path.

inline std::string device(const std::string& dev) {
    return "total@" + dev;
}

// The key is the _real_ (prototype) submodel id, so all calls of one repeating block
// accumulate under a single entry.
inline std::string submodel(const std::string& dev, std::size_t real_idx, bool is_funcall) {
    std::ostringstream ss;
    ss << device(dev) << "/submodel[" << std::setw(3) << std::setfill('0') << real_idx << "]"
       << (is_funcall ? "(fn)" : "");
    return ss.str();
}

inline std::string pyramid(const std::string& submodel_tag, std::size_t pyramid_id, std::size_t context_length) {
    std::ostringstream ss;
    ss << submodel_tag << "/attn/pyramid[" << std::setw(2) << std::setfill('0') << pyramid_id
       << "] kv=" << context_length;
    return ss.str();
}

inline std::string hfa_tile(const std::string& submodel_tag, int64_t tile_size) {
    return submodel_tag + "/attn/hfa/tile[" + std::to_string(tile_size) + "]";
}

inline std::string hfa_final_tile(const std::string& submodel_tag) {
    return submodel_tag + "/attn/hfa/tile[final]";
}

inline std::string hfa_host_prep(const std::string& submodel_tag) {
    return submodel_tag + "/attn/hfa/host-prep";
}

}  // namespace tags
}  // namespace perf
}  // namespace npuw
}  // namespace ov
