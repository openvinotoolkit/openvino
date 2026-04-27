// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TEMPORARY DEBUG INSTRUMENTATION — for investigating the flaky
// GRUFusionTest.GRUCellPattern case on macOS arm64.
//
// WHAT:  small helpers that print parameter lists / pattern-map bindings / port
//        shape maps to stderr when env var OV_TMP_DEBUG=1 is set.
// HOW:   static inline functions; each caller site wraps its dump in
//        `if (ov::tmp_debug::enabled())`.  No runtime cost when env var is
//        unset (single getenv read per process, cached in a static bool).
// WHY:   the flake is reproducible only on macOS arm64.  We need
//        instrumentation that can be enabled without rebuild in CI (env var),
//        writes to stderr so the existing gtest logging picks it up, and is
//        cheap enough to leave on for many iterations while collecting data.
//
// To be reverted once the root cause is understood.
#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace ov {
namespace tmp_debug {

inline bool enabled() {
    static const bool on = []() {
        const char* e = std::getenv("OV_TMP_DEBUG");
        return e && e[0] && std::string(e) != "0";
    }();
    return on;
}

inline std::ostream& log() {
    return std::cerr << "[OV_TMP_DEBUG] ";
}

template <typename ParamVector>
inline void dump_params(const char* tag, const ParamVector& params) {
    if (!enabled())
        return;
    log() << "-- params dump: " << tag << " (count=" << params.size() << ") --\n";
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& p = params[i];
        std::ostringstream shape_s;
        shape_s << p->get_partial_shape();
        std::cerr << "[OV_TMP_DEBUG]   i=" << i << "  ptr=" << static_cast<const void*>(p.get())
                  << "  shape=" << shape_s.str() << "  name=" << p->get_friendly_name() << "\n";
    }
}

}  // namespace tmp_debug
}  // namespace ov
