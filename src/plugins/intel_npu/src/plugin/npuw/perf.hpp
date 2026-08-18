// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "logging.hpp"

namespace ov {
namespace npuw {
namespace perf {

// Whether the instrumentation is compiled in at all. In a non-debug-caps build the
// whole framework degrades to empty types with constexpr no-op methods, so neither
// the profile members nor the call sites cost anything (see NoProfile/NoMetric below).
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
constexpr bool compiled_in = true;
#else
constexpr bool compiled_in = false;
#endif

class format_guard {
public:
    format_guard(std::ostream& os) : m_state(&m_sbuf), m_os(os) {
        m_state.copyfmt(m_os);
    }

    ~format_guard() {
        m_os.copyfmt(m_state);
    }

private:
    struct : std::streambuf {
    } m_sbuf;
    std::ios m_state;
    std::ostream& m_os;
};

// Templated on purpose: no std::function is constructed on the measured path.
template <typename F>
float ms_to_run(F&& body) {
    namespace khr = std::chrono;
    const auto s = khr::steady_clock::now();
    body();
    const auto f = khr::steady_clock::now();
    return khr::duration_cast<khr::microseconds>(f - s).count() / 1000.0f;
}

// What a profile measures - decides whether a run boundary may drop its samples.
enum class Scope {
    Compilation,  // compile-time statistics, never reset by a run
    Execution,    // per-run statistics, reset at a run boundary
    Lifetime      // request-lifetime statistics (memory), never reset by a run
};

// Profiles register themselves here so a run boundary can scope the collected statistics
// to a single run (e.g. one conversation) instead of the whole process lifetime.
class IProfile {
public:
    virtual ~IProfile() = default;
    virtual Scope scope() const = 0;
    virtual void reset() = 0;
    // run_id is supplied by the boundary, so every Execution profile labels the same
    // conversation with the same number even if it collected nothing during it.
    virtual void snapshot_and_reset(std::size_t run_id) = 0;
    virtual bool has_samples() const = 0;
};

void register_profile(IProfile* profile);
void unregister_profile(IProfile* profile);

// Run-boundary operations. Only profiles of the matching scope are affected.
void reset_all(Scope scope);
void snapshot_and_reset_all(Scope scope);

// Id of the run currently being collected - shared by all profiles
std::size_t current_run_index();

// How many past runs are kept as snapshots (OPENVINO_NPUW_PROF_KEEP_RUNS, default 8)
std::size_t keep_runs();

// Profiling is supported for a single NPUW pipeline at a time - see the contract in
// docs. Registering a second run-boundary owner logs a one-time warning.
void register_run_boundary_owner(const void* owner);
void unregister_run_boundary_owner(const void* owner);

struct MSec {
    constexpr static const char* name = "ms";
    constexpr static bool timing = true;

    using type_t = float;

    template <typename F>
    static type_t sample(F&& body) {
        return ms_to_run(std::forward<F>(body));
    }
};

struct Bytes {
    constexpr static const char* name = "MB";
    constexpr static bool timing = false;
    constexpr static float scaler = 1024.0 * 1024;

    using type_t = uint64_t;
};

namespace detail {

// Prints "<label> [ ... ]" with the label padded to the given width
inline void print_label(std::ostream& os, const std::string& label, int width) {
    os << std::left << std::setw(std::max(10, width)) << label;
}

template <typename U>
class Metric {
    using T = typename U::type_t;

private:
    std::vector<T> records;
    T vmin = std::numeric_limits<T>::max();
    T vmax = std::numeric_limits<T>::lowest();
    T total = T(0);
    std::string name;
    bool enabled = false;

public:
    using unit_t = U;
    using value_t = T;

    Metric() = default;
    Metric(Metric&& m)
        : records(std::move(m.records)),
          vmin(m.vmin),
          vmax(m.vmax),
          total(m.total),
          name(std::move(m.name)),
          enabled(m.enabled) {}
    Metric(const Metric&) = delete;
    Metric& operator=(const Metric&) = delete;

    explicit Metric(const std::string& named, bool active = false) : name(named), enabled(active) {}

    void enable() {
        enabled = true;
    }

    bool is_enabled() const {
        return enabled;
    }

    const std::string& tag() const {
        return name;
    }

    // NB: T is taken by value on purpose - an rvalue reference parameter breaks on MSVC,
    // which treats the functional cast float(x) as an lvalue (C2679).
    void operator+=(T t) {
        if (!enabled) {
            return;
        }
        if (records.empty()) {
            records.reserve(64);
        }
        vmin = std::min(vmin, t);
        vmax = std::max(vmax, t);
        total += t;
        records.push_back(t);
    }

    // Non-templated, non-type-erased entry point for the measured path
    void add(T t) {
        *this += t;
    }

    bool has_samples() const {
        return !records.empty();
    }

    std::size_t count() const {
        return records.size();
    }

    T sum() const {
        return total;
    }

    float sum_f() const {
        return static_cast<float>(total);
    }

    // Test-only: inspect the collected samples before any formatting happens
    const std::vector<T>& samples() const {
        return records;
    }

    void reset() {
        records.clear();
        vmin = std::numeric_limits<T>::max();
        vmax = std::numeric_limits<T>::lowest();
        total = T(0);
    }

    // Moves the collected samples out, leaving this metric live but empty
    Metric take() {
        Metric out(name, enabled);
        out.records = std::move(records);
        out.vmin = vmin;
        out.vmax = vmax;
        out.total = total;
        reset();
        return out;
    }

    float avg() const {
        NPUW_ASSERT(enabled);
        if (records.empty()) {
            return 0.f;
        }
        return static_cast<float>(total) / static_cast<float>(records.size());
    }

    T med() const {
        NPUW_ASSERT(enabled);
        std::vector<T> cpy(records);
        std::nth_element(cpy.begin(), cpy.begin() + cpy.size() / 2, cpy.end());
        return cpy[cpy.size() / 2];
    }

    template <typename F>
    void record(F&& f) {
        if (!enabled) {
            f();
        } else {
            *this += U::sample(std::forward<F>(f));
        }
    }

    void print(std::ostream& os, const std::string& label, int width = 44) const {
        format_guard fmt(os);

        const char* units = U::name;
        print_label(os, label.empty() ? std::string("<unnamed timer>") : label, width);
        if (!enabled) {
            os << "[ disabled ]";
            return;
        }
        if (records.empty()) {
            os << "[ no data ]";
            return;
        }
        os << "[ avg = " << avg() << " " << units << ", med = " << med() << " " << units << " in " << vmin << ".."
           << vmax << " " << units << " range over " << records.size() << " records"
           << ", total = " << total << units << " ]";
        if (ov::npuw::profiling_details()) {
            for (std::size_t i = 0u; i < records.size(); i++) {
                os << (i % 10 == 0 ? "\n      " : " ") << records[i];
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Metric<U>& m) {
        m.print(os, m.name);
        return os;
    }
};

template <typename U>
class Counter {
public:
    using unit_t = U;
    using T = typename U::type_t;
    using value_t = T;

private:
    T total = 0;
    std::size_t calls = 0;
    std::string name;
    bool enabled = false;

public:
    Counter() = default;
    Counter(Counter&& c) : total(c.total), calls(c.calls), name(std::move(c.name)), enabled(c.enabled) {}
    Counter(const Counter&) = delete;
    Counter& operator=(const Counter&) = delete;

    explicit Counter(const std::string& named, bool active = false) : name(named), enabled(active) {}

    void enable() {
        enabled = true;
    }

    bool is_enabled() const {
        return enabled;
    }

    const std::string& tag() const {
        return name;
    }

    // NB: by value - see the note in Metric::operator+=
    void operator+=(T t) {
        if (enabled) {
            total += t;
            calls++;
        }
    }

    void add(T t) {
        *this += t;
    }

    bool has_samples() const {
        return calls != 0u;
    }

    std::size_t count() const {
        return calls;
    }

    T sum() const {
        return total;
    }

    float sum_f() const {
        return static_cast<float>(total) / U::scaler;
    }

    void reset() {
        total = 0;
        calls = 0;
    }

    Counter take() {
        Counter out(name, enabled);
        out.total = total;
        out.calls = calls;
        reset();
        return out;
    }

    void print(std::ostream& os, const std::string& label, int width = 44) const {
        format_guard fmt(os);

        const char* units = U::name;
        print_label(os, label.empty() ? std::string("<unnamed counter>") : label, width);
        if (!enabled) {
            os << "[ disabled ]";
            return;
        }
        os << "[ " << sum_f() << " " << units << " total in " << calls << " records ]";
    }

    friend std::ostream& operator<<(std::ostream& os, const Counter<U>& c) {
        c.print(os, c.name);
        return os;
    }
};

// Disabled-build counterparts. Empty types with constexpr no-op methods, so a
// non-debug-caps build has neither storage nor call-site code.
template <typename U>
class NoMetric {
public:
    using unit_t = U;
    using T = typename U::type_t;
    using value_t = T;

    constexpr NoMetric() = default;
    constexpr explicit NoMetric(const std::string&, bool = false) {}

    constexpr void enable() const noexcept {}
    constexpr bool is_enabled() const noexcept {
        return false;
    }
    constexpr void operator+=(T) const noexcept {}
    constexpr void add(T) const noexcept {}
    constexpr bool has_samples() const noexcept {
        return false;
    }
    constexpr std::size_t count() const noexcept {
        return 0u;
    }
    constexpr T sum() const noexcept {
        return T(0);
    }
    constexpr float sum_f() const noexcept {
        return 0.f;
    }
    constexpr void reset() const noexcept {}

    template <typename F>
    void record(F&& f) const {
        f();
    }
};

// Swallows any assignment - used for the fields of a compiled-out profile
struct sink {
    template <typename T>
    constexpr const sink& operator=(T&&) const noexcept {
        return *this;
    }
};

template <class M>
struct NoProfile {
    using metric_t = M;

    sink area;
    sink report_on_die;
    sink profile_scope;
    sink children_overlap;

    NoProfile() = default;
    NoProfile(const NoProfile&) = delete;
    NoProfile& operator=(const NoProfile&) = delete;

    M& operator[](const std::string&) const {
        static M dummy;
        return dummy;
    }
    constexpr M* handle(const std::string&) const noexcept {
        return nullptr;
    }
    template <typename F>
    void record(const char*, F&& f) const {
        f();
    }
    constexpr void reset() const noexcept {}
    constexpr void snapshot_and_reset() const noexcept {}
    constexpr void snapshot_and_reset(std::size_t) const noexcept {}
    constexpr bool has_samples() const noexcept {
        return false;
    }
    void report(std::ostream& = std::cout) const noexcept {}
};

template <class M>
struct RealProfile : public IProfile {
    using metric_t = M;

    struct Snapshot {
        std::size_t index = 0u;
        std::map<std::string, M> metrics;

        // Metrics are move-only, so the implicit copy ctor is ill-formed. It is still
        // *declared*, which is enough for MSVC's vector reallocation to pick the copy
        // path over the (non-noexcept) map move and then fail to instantiate it.
        Snapshot() = default;
        Snapshot(Snapshot&&) = default;
        Snapshot& operator=(Snapshot&&) = default;
        Snapshot(const Snapshot&) = delete;
        Snapshot& operator=(const Snapshot&) = delete;
    };

    std::map<std::string, M> metrics;
    std::vector<Snapshot> snapshots;
    std::string area;
    bool report_on_die = false;
    Scope profile_scope = Scope::Execution;
    // Set when sibling entries of the top level may genuinely overlap in time
    // (async funcall pipelining) - then no residual row is meaningful there.
    bool children_overlap = false;

    RealProfile() {
        register_profile(this);
        // A profile created mid-session (e.g. a lazily compiled generate variant) must
        // still label its first run with the id everybody else is on
        m_run_index = current_run_index();
    }
    RealProfile(const RealProfile&) = delete;
    RealProfile& operator=(const RealProfile&) = delete;

    Scope scope() const override {
        return profile_scope;
    }

    M& operator[](const std::string& tag) {
        return ensure(tag);
    }

    // Resolves a tag ONCE. Returns nullptr when the profile is disabled, so callers
    // can skip both the lookup and the timing. Mapped-value references of a std::map
    // are stable across inserts and across reset(), so the handle stays valid for
    // the profile's lifetime.
    M* handle(const std::string& tag) {
        if (!report_on_die) {
            return nullptr;
        }
        return &ensure(tag);
    }

    // Gated stage timing for call sites that cannot pre-resolve a handle. When the
    // profile is disabled the body still runs, but no tag string is built and no
    // map lookup happens.
    template <typename F>
    void record(const char* tag, F&& f) {
        if (!report_on_die) {
            f();
            return;
        }
        ensure(tag).record(std::forward<F>(f));
    }

    // Keeps the tags but drops the collected samples
    void reset() override {
        for (auto&& m : metrics) {
            m.second.reset();
        }
    }

    // Run boundary: move the live samples into a snapshot. No formatting, no IO -
    // everything is printed once, at destruction. run_id comes from the boundary so
    // that all Execution profiles agree on the numbering even when one of them was
    // inactive for a whole conversation.
    void snapshot_and_reset(std::size_t run_id) override {
        m_run_index = run_id + 1u;
        if (!has_samples()) {
            return;
        }
        Snapshot s;
        s.index = run_id;
        for (auto&& m : metrics) {
            if (m.second.has_samples()) {
                s.metrics.emplace(m.first, m.second.take());
            }
        }
        snapshots.push_back(std::move(s));
        const auto cap = keep_runs();
        while (snapshots.size() > cap) {
            snapshots.erase(snapshots.begin());
        }
    }

    void snapshot_and_reset() {
        snapshot_and_reset(m_run_index);
    }

    bool has_samples() const override {
        for (auto&& m : metrics) {
            if (m.second.has_samples()) {
                return true;
            }
        }
        return false;
    }

    void report(std::ostream& os = std::cout) const {
        if (snapshots.empty() && !has_samples()) {
            return;
        }

        format_guard fmt(os);

        if (!area.empty()) {
            os << area << ":" << std::endl;
        } else {
            os << std::hex << this << ":" << std::endl;
        }
        os << "  [ entries are nested - never sum across levels;"
           << " with async funcall pipelining sibling intervals may overlap ]" << std::endl;
        for (auto&& s : snapshots) {
            os << "  run[" << s.index << "]:" << std::endl;
            report_tree(os, s.metrics, children_overlap);
        }
        if (has_samples()) {
            os << "  run[" << m_run_index << "] (current):" << std::endl;
            report_tree(os, metrics, children_overlap);
        }
    }

    ~RealProfile() {
        unregister_profile(this);
        if (report_on_die) {
            try {
                // avg() & others may throw exceptions if
                // enabled behavior is broken
                report();
            } catch (...) {
            }
        }
    }

private:
    std::size_t m_run_index = 0u;
#ifndef NDEBUG
    // I1: a profile is only ever written by the thread that executes the corresponding
    // subgraph. Caught here rather than by silent map corruption later.
    mutable std::thread::id m_owner;
#endif

    void check_owner() const {
#ifndef NDEBUG
        if (profile_scope != Scope::Execution || !report_on_die) {
            return;
        }
        const auto self = std::this_thread::get_id();
        if (m_owner == std::thread::id()) {
            m_owner = self;
            return;
        }
        NPUW_ASSERT(m_owner == self && "NPUW perf: profile accessed from a foreign thread");
#endif
    }

    M& ensure(const std::string& tag) {
        check_owner();
        auto iter = metrics.find(tag);
        if (iter == metrics.end()) {
            // Use report_on_die as an "enabled" marker here
            auto [new_iter, unused] = metrics.insert({tag, M(tag, report_on_die)});
            return new_iter->second;
        }
        return iter->second;
    }

    // The tags are '/'-separated paths, so the map's own ordering is already the tree
    // order (pre-order): a child sorts right after its parent, siblings sort together.
    static void report_tree(std::ostream& os, const std::map<std::string, M>& ms, bool overlap_hint) {
        struct Row {
            const std::string* tag;
            const M* metric;
            std::string display;
            int indent;
            float total;
            int parent;  // index into rows, -1 for a root
            bool alias;
        };

        auto is_child_of = [](const std::string& parent, const std::string& tag) {
            return tag.size() > parent.size() + 1 && tag.compare(0, parent.size(), parent) == 0 &&
                   tag[parent.size()] == '/';
        };
        auto names_a_level = [](const std::string& display) {
            return display.rfind("attn/pyramid[", 0) == 0;
        };

        // Pass 1: flatten into rows, linking each row to its nearest present ancestor
        std::vector<Row> rows;
        std::vector<int> stack;
        for (const auto& kv : ms) {
            if (!kv.second.has_samples()) {
                // Resolved but unused this run (e.g. a pyramid level that was not selected)
                continue;
            }
            const std::string& tag = kv.first;
            while (!stack.empty() && !is_child_of(*rows[stack.back()].tag, tag)) {
                stack.pop_back();
            }
            const int parent = stack.empty() ? -1 : stack.back();
            const int indent = parent < 0 ? 0 : rows[parent].indent + 1;
            std::string display = parent < 0 ? tag : tag.substr(rows[parent].tag->size() + 1);
            const bool alias = names_a_level(display);
            rows.push_back(Row{&tag, &kv.second, std::move(display), indent, kv.second.sum_f(), parent, alias});
            stack.push_back(static_cast<int>(rows.size()) - 1);
        }

        // Pass 2: per-parent child statistics
        std::vector<std::size_t> children(rows.size(), 0u);
        std::vector<std::size_t> alias_children(rows.size(), 0u);
        std::vector<float> child_sum(rows.size(), 0.f);
        for (const auto& r : rows) {
            if (r.parent < 0) {
                continue;
            }
            children[r.parent]++;
            child_sum[r.parent] += r.total;
            if (r.alias) {
                alias_children[r.parent]++;
            }
        }
        // A pyramid level is an alias of the whole subgraph run only when it is the
        // parent's only child. Several levels selected within one run (chunked prefill)
        // partition the parent's calls instead, so their totals must be summed.
        auto is_sole_alias = [&](std::size_t i) {
            return rows[i].alias && rows[i].parent >= 0 && children[rows[i].parent] == 1u;
        };

        auto pad = [](int indent) {
            return std::string(2 + indent * 2, ' ');
        };
        auto emit_residual = [&](std::size_t i) {
            if (!U_is_timing() || children[i] == 0u) {
                return;
            }
            if (children[i] == 1u && alias_children[i] == 1u) {
                return;  // the single child is the parent, under another name
            }
            const int indent = rows[i].indent + 1;
            if (overlap_hint && rows[i].indent == 0) {
                os << pad(indent);
                print_label(os, "(overlapped)", 44 - indent * 2);
                os << "[ children run concurrently - residual is not meaningful ]" << std::endl;
                return;
            }
            const float residual = rows[i].total - child_sum[i];
            const float tol = std::max(0.05f, 0.001f * std::fabs(rows[i].total));
            if (residual > tol) {
                os << pad(indent);
                print_label(os, "(unattributed)", 44 - indent * 2);
                os << "[ total = " << residual << " ms ]" << std::endl;
            } else if (residual < -tol) {
                os << pad(indent);
                print_label(os, "(overlapped)", 44 - indent * 2);
                os << "[ children overlap - residual is not meaningful ]" << std::endl;
            }
            // within tolerance: timing noise, nothing to report
        };

        // Pass 3: print, emitting a parent's residual after its last descendant
        std::vector<int> open;
        for (std::size_t i = 0; i < rows.size(); i++) {
            while (!open.empty() && open.back() != rows[i].parent) {
                emit_residual(static_cast<std::size_t>(open.back()));
                open.pop_back();
            }
            os << pad(rows[i].indent);
            rows[i].metric->print(os, rows[i].display, 44 - rows[i].indent * 2);
            if (is_sole_alias(i)) {
                os << " (= parent)";
            }
            os << std::endl;
            open.push_back(static_cast<int>(i));
        }
        while (!open.empty()) {
            emit_residual(static_cast<std::size_t>(open.back()));
            open.pop_back();
        }
    }

    static constexpr bool U_is_timing() {
        return M::unit_t::timing;
    }
};

// Per-index handle caches. Empty in a non-debug-caps build.
template <class M>
struct RealHandles {
    std::vector<M*> v;

    void resize(std::size_t n) {
        v.assign(n, nullptr);
    }
    void clear() {
        std::fill(v.begin(), v.end(), nullptr);
    }
    void set(std::size_t i, M* m) {
        v[i] = m;
    }
    M* get(std::size_t i) const {
        return v[i];
    }
    bool empty() const {
        return v.empty();
    }
};

template <class M>
struct NoHandles {
    constexpr void resize(std::size_t) const noexcept {}
    constexpr void clear() const noexcept {}
    constexpr void set(std::size_t, M*) const noexcept {}
    constexpr M* get(std::size_t) const noexcept {
        return nullptr;
    }
    constexpr bool empty() const noexcept {
        return true;
    }
};

// Per-index tag caches - a tag string is built at most once per index, never on
// the measured path. Empty in a non-debug-caps build.
struct RealTags {
    std::vector<std::string> v;

    void resize(std::size_t n) {
        v.resize(n);
    }
    std::string& at(std::size_t i) {
        return v[i];
    }
};

struct NoTags {
    constexpr void resize(std::size_t) const noexcept {}
    std::string& at(std::size_t) const {
        static std::string none;
        return none;
    }
};

}  // namespace detail

template <typename U>
using metric = std::conditional_t<compiled_in, detail::Metric<U>, detail::NoMetric<U>>;

template <typename U>
using counter = std::conditional_t<compiled_in, detail::Counter<U>, detail::NoMetric<U>>;

template <class M>
using Profile = std::conditional_t<compiled_in, detail::RealProfile<M>, detail::NoProfile<M>>;

template <class M>
using handles = std::conditional_t<compiled_in, detail::RealHandles<M>, detail::NoHandles<M>>;

using tag_cache = std::conditional_t<compiled_in, detail::RealTags, detail::NoTags>;

using ms_metric = metric<MSec>;
using metric_ptr = ms_metric*;
using ms_handles = handles<ms_metric>;

namespace detail {

// Allocation-free RAII timer. A null handle makes it inert: no clock is read at all,
// which is the one predictable branch the disabled path is allowed to pay.
class RealSample {
public:
    explicit RealSample(ms_metric* m) noexcept
        : m_metric(m),
          m_start(m ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}

    RealSample(const RealSample&) = delete;
    RealSample& operator=(const RealSample&) = delete;
    RealSample(RealSample&&) = delete;
    RealSample& operator=(RealSample&&) = delete;

    float elapsed_ms() const noexcept {
        if (!m_metric) {
            return 0.f;
        }
        if (m_stopped) {
            return m_elapsed;
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start)
                   .count() /
               1000.0f;
    }

    // Stops the timer and marks the sample for commit. Must be reached explicitly:
    // a body that throws unwinds past it, so a failed run adds no sample and the
    // record count stays "number of successful runs".
    float commit() noexcept {
        if (!m_metric) {
            return 0.f;
        }
        if (!m_stopped) {
            m_elapsed = elapsed_ms();
            m_stopped = true;
        }
        m_commit = true;
        return m_elapsed;
    }

    void discard() noexcept {
        m_commit = false;
    }

    ~RealSample() noexcept {
        if (m_commit && m_metric) {
            // Recording allocates; a failure here must not turn optional profiling
            // into a std::terminate out of a noexcept destructor.
            try {
                *m_metric += m_elapsed;
            } catch (...) {
            }
        }
    }

private:
    ms_metric* m_metric = nullptr;
    std::chrono::steady_clock::time_point m_start;
    float m_elapsed = 0.f;
    bool m_stopped = false;
    bool m_commit = false;
};

class NoSample {
public:
    constexpr explicit NoSample(ms_metric*) noexcept {}

    NoSample(const NoSample&) = delete;
    NoSample& operator=(const NoSample&) = delete;
    NoSample(NoSample&&) = delete;
    NoSample& operator=(NoSample&&) = delete;

    constexpr float elapsed_ms() const noexcept {
        return 0.f;
    }
    constexpr float commit() const noexcept {
        return 0.f;
    }
    constexpr void discard() const noexcept {}
};

}  // namespace detail

using sample = std::conditional_t<compiled_in, detail::RealSample, detail::NoSample>;

}  // namespace perf
}  // namespace npuw
}  // namespace ov
