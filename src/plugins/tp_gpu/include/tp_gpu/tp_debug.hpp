// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <ostream>
#include <sstream>

#include "openvino/core/log_util.hpp"

namespace ov {
namespace tp_gpu {

/// Where the plugin's diagnostics go.
///
/// Inline on purpose: the collective primitives live in the intel_gpu plugin,
/// which does not link the TP_GPU plugin module, so anything they call has to
/// be header-only.
inline std::ostream& log_stream() {
    return std::cerr;
}

/// Collects one message and hands it to the core logger on destruction.
/// Used for the few warnings that report a real configuration problem and so
/// have to reach the user in a release build, where the debug macros below
/// compile away.
class AlwaysWarn {
public:
    ~AlwaysWarn() {
        ov::util::log_message(m_message.str());
    }
    std::ostringstream& stream() {
        return m_message;
    }

private:
    std::ostringstream m_message;
};

}  // namespace tp_gpu
}  // namespace ov

// ---------------------------------------------------------------------------
// Call-site macros.
//
// Everything here keys off TP_DEBUG_CONFIG, which CMake defines from
// ENABLE_TP_GPU_DEBUG_CAPS -- one switch for the whole plugin.  The option
// table in options.inl uses the same switch, so a debug option either exists
// everywhere or nowhere; there is no build in which one is settable and
// silently ignored.
//
// Without it the debug options have no getters at all, so every use of one has
// to go through TP_DEBUG_OPT, which substitutes the release-build value
// instead of touching the config.  The conditions then fold to constants and
// the guarded code disappears with them.
//
// Note: the macros below name ov::tp_gpu::TPConfig, so a translation unit that
// uses them has to include "tp_gpu/tp_config.hpp" (which includes this file).
// ---------------------------------------------------------------------------

#ifdef TP_DEBUG_CONFIG

#    define TP_DEBUG_IF(cond)  if (cond)
#    define TP_DEBUG_CODE(...) __VA_ARGS__

/// The value of a per-model debug option, or `release_value` where the option
/// does not exist.  `config` is any TPConfig, `option` the bare property name.
#    define TP_DEBUG_OPT(config, option, release_value) ((config).get_##option())

/// The same for a global debug option, which has a static getter and no
/// config object.
#    define TP_DEBUG_GLOBAL_OPT(option, release_value) (ov::tp_gpu::TPConfig::get_##option())

#    define TP_LOG_RAW(level) \
        if (ov::tp_gpu::TPConfig::verbosity() >= (level)) ov::tp_gpu::log_stream()

#else

#    define TP_DEBUG_IF(cond)  if (false)
#    define TP_DEBUG_CODE(...)
#    define TP_DEBUG_OPT(config, option, release_value)  (release_value)
#    define TP_DEBUG_GLOBAL_OPT(option, release_value)   (release_value)
#    define TP_LOG_RAW(level) \
        if (false) ov::tp_gpu::log_stream()

#endif

/// Whether the requested verbosity is in effect.  Reads the process-global
/// option, so it works from code that has no config object at hand -- which is
/// the case for the collectives living inside the intel_gpu plugin.
#define TP_VERBOSE_AT_LEAST(level) (ov::tp_gpu::TPConfig::verbosity() >= (level))

/// Errors that must never be silent, down to per-call tracing.  `TP_LOG_ERR`
/// and `TP_LOG_WARN` are for things a user needs to see; the rest are for
/// whoever is debugging the plugin.
#define TP_LOG_ERR   TP_LOG_RAW(ov::log::Level::ERR)
#define TP_LOG_WARN  TP_LOG_RAW(ov::log::Level::WARNING)
#define TP_LOG_INFO  TP_LOG_RAW(ov::log::Level::INFO)
#define TP_LOG_DEBUG TP_LOG_RAW(ov::log::Level::DEBUG)
#define TP_LOG_TRACE TP_LOG_RAW(ov::log::Level::TRACE)

/// A warning the user sees in every build type.  Not gated by the verbosity:
/// these report a real configuration problem, not a diagnostic.
#define TP_WARN_ALWAYS ov::tp_gpu::AlwaysWarn{}.stream()

/// Measurement output.  Printed whenever profiling is on, whatever the
/// verbosity, because asking for measurements and getting silence would be
/// surprising.
#define TP_REPORT ov::tp_gpu::log_stream()
