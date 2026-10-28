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

namespace cldnn {
namespace ze {
namespace tp {

/// Where the collectives' diagnostics and measurement reports go.
inline std::ostream& log_stream() {
    return std::cout;
}

/// Collects one message and hands it to the core logger on destruction. For the
/// few warnings that report a real configuration problem and so have to reach
/// the user whatever the verbosity.
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

/// Adds the time its scope took to `into`, an accumulator with `add(uint64_t ns)`.
/// With `enabled` false the clock is never read.
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

}  // namespace tp
}  // namespace ze
}  // namespace cldnn

/// A warning the user sees whatever the verbosity: a real configuration problem, not a diagnostic.
#define TP_WARN_ALWAYS ::cldnn::ze::tp::AlwaysWarn{}.stream()

/// Measurement output. Printed whenever profiling is on, whatever the verbosity.
#define TP_REPORT ::cldnn::ze::tp::log_stream()
