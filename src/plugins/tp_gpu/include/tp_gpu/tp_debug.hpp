// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <sstream>

#include "openvino/core/log_util.hpp"

namespace ov {
namespace tp_gpu {

inline std::ostream& log_stream() {
    return std::cout;
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

/// Nanoseconds gathered from one or more scopes.
struct NS {
    uint64_t value{0};

    void add(uint64_t ns) {
        value += ns;
    }
    uint64_t ns() const {
        return value;
    }
    double us() const {
        return static_cast<double>(value) / 1e3;
    }
    double ms() const {
        return static_cast<double>(value) / 1e6;
    }
};

/// Time since construction, read explicitly.
///
/// For the measurements a scope does not line up with: one start, several
/// readings, or a reading taken from somewhere the start is not visible.
class Stopwatch {
public:
    explicit Stopwatch(bool enabled)
        : m_enabled(enabled),
          m_start(enabled ? Clock::now() : Clock::time_point{}) {}

    /// Zero when disabled, so a caller never has to ask.
    NS elapsed() const {
        if (!m_enabled) {
            return {};
        }
        return NS{static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_start).count())};
    }

private:
    using Clock = std::chrono::steady_clock;

    bool m_enabled;
    Clock::time_point m_start;
};

/// Adds the time its scope took to `into`.
///
/// Takes any accumulator with `add(uint64_t ns)`: the plain NS above, or an
/// atomic counter where several ranks write at once. `enabled` reaches every
/// call site as a constant -- it comes from a config option that folds to a
/// literal in a build without debug caps -- so with profiling off the clock
/// reads disappear along with the object.
template <typename Accumulator>
class ScopedTime {
public:
    ScopedTime(bool enabled, Accumulator& into) : m_watch(enabled), m_into(enabled ? &into : nullptr) {}

    ~ScopedTime() {
        if (m_into != nullptr) {
            m_into->add(m_watch.elapsed().ns());
        }
    }

    ScopedTime(const ScopedTime&) = delete;
    ScopedTime& operator=(const ScopedTime&) = delete;

private:
    Stopwatch m_watch;
    Accumulator* m_into;
};

template <typename Accumulator>
ScopedTime(bool, Accumulator&) -> ScopedTime<Accumulator>;

/// Counts calls and says when the next report is due.
class DumpPeriod {
public:
    explicit DumpPeriod(std::size_t period) : m_period(std::max<std::size_t>(std::size_t{1}, period)) {}

    /// Records one call. True once every `period` of them.
    bool due() {
        ++m_calls;
        if (m_calls - m_last < m_period) {
            return false;
        }
        m_last = m_calls;
        return true;
    }

    uint64_t calls() const {
        return m_calls;
    }

    std::size_t period() const {
        return m_period;
    }

private:
    std::size_t m_period;
    uint64_t m_calls{0};
    uint64_t m_last{0};
};

}  // namespace tp_gpu
}  // namespace ov

// ---------------------------------------------------------------------------
// Call-site macros.
//
// Everything here keys off TP_DEBUG_CONFIG, which CMake defines from
// ENABLE_TP_GPU_DEBUG_CAPS -- one switch for the whole plugin. The option
// table in options.inl uses the same switch, so a debug option either exists
// everywhere or nowhere; there is no build in which one is settable and
// silently ignored.
//
// Without it the debug options have no getters at all, so every use of one has
// to go through TP_DEBUG_OPT, which substitutes the release-build value
// instead of touching the config. The conditions then fold to constants and
// the guarded code disappears with them.
//
// Note: the macros below name ov::tp_gpu::TPConfig, so a translation unit that
// uses them has to include "tp_gpu/tp_config.hpp" (which includes this file).
// ---------------------------------------------------------------------------

#ifdef TP_DEBUG_CONFIG

#    define TP_DEBUG_IF(cond)  if (cond)
#    define TP_DEBUG_CODE(...) __VA_ARGS__

/// The value of a per-model debug option, or `release_value` where the option
/// does not exist. `config` is any TPConfig, `option` the bare property name.
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

/// Whether the requested verbosity is in effect. Reads the process-global
/// option, so it works from code that has no config object at hand -- which is
/// the case for the collectives living inside the intel_gpu plugin.
#define TP_VERBOSE_AT_LEAST(level) (ov::tp_gpu::TPConfig::verbosity() >= (level))

/// Errors that must never be silent, down to per-call tracing. `TP_LOG_ERR`
/// and `TP_LOG_WARN` are for things a user needs to see; the rest are for
/// whoever is debugging the plugin.
#define TP_LOG_ERR   TP_LOG_RAW(ov::log::Level::ERR)
#define TP_LOG_WARN  TP_LOG_RAW(ov::log::Level::WARNING)
#define TP_LOG_INFO  TP_LOG_RAW(ov::log::Level::INFO)
#define TP_LOG_DEBUG TP_LOG_RAW(ov::log::Level::DEBUG)
#define TP_LOG_TRACE TP_LOG_RAW(ov::log::Level::TRACE)

/// A warning the user sees in every build type. Not gated by the verbosity:
/// these report a real configuration problem, not a diagnostic.
#define TP_WARN_ALWAYS ov::tp_gpu::AlwaysWarn{}.stream()

/// Measurement output. Printed whenever profiling is on, whatever the verbosity.
#define TP_REPORT ov::tp_gpu::log_stream()
