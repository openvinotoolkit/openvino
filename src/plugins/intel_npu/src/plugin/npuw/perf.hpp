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
#include <utility>
#include <vector>

#include "logging.hpp"

namespace ov {
namespace npuw {
namespace perf {

// Whether the instrumentation is compiled in at all. In a non-debug-caps build the
// types stay (small per-object footprint), but the hot methods below fold to the
// plain body at compile time. Helpers check the flag before resolving tags.
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
constexpr bool compiled_in = true;
#else
constexpr bool compiled_in = false;
#endif

inline bool enabled_by_default() {
    return compiled_in && ov::npuw::profiling_enabled();
}

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

// What a profile measures - decides whether a run boundary may drop its samples.
enum class Scope {
    Compilation,  // compile-time statistics, never reset by a run
    Execution,    // per-run statistics, reset at a run boundary
    Lifetime      // request-lifetime statistics (memory), never reset by a run
};

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

// Each LLM request owns its boundary.
class RunBoundary {
public:
    std::size_t current_run_index() const {
        return m_run_index;
    }

    void snapshot_and_reset(const std::vector<IProfile*>& profiles) {
        const auto ending = m_run_index++;
        for (auto* profile : profiles) {
            profile->snapshot_and_reset(ending);
        }
    }

private:
    std::size_t m_run_index = 0u;
};

// How many past runs are kept as snapshots (OPENVINO_NPUW_PROF_KEEP_RUNS, default 8)
std::size_t keep_runs();

struct MSec {
    constexpr static const char* name = "ms";
    constexpr static bool timing = true;

    using type_t = float;
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

}  // namespace detail

template <typename U>
class metric {
    using T = typename U::type_t;

private:
    std::vector<T> records;
    T vmin = std::numeric_limits<T>::max();
    T vmax = std::numeric_limits<T>::lowest();
    T total = T(0);

public:
    using unit_t = U;
    using value_t = T;

    metric() = default;
    metric(metric&& m) : records(std::move(m.records)), vmin(m.vmin), vmax(m.vmax), total(m.total) {}
    metric(const metric&) = delete;
    metric& operator=(const metric&) = delete;

    // NB: T is taken by value on purpose - an rvalue reference parameter breaks on MSVC,
    // which treats the functional cast float(x) as an lvalue (C2679).
    void operator+=(T t) {
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
        if constexpr (U::timing) {
            return static_cast<float>(total);
        } else {
            return static_cast<float>(total) / U::scaler;
        }
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
    metric take() {
        metric out;
        out.records = std::move(records);
        out.vmin = vmin;
        out.vmax = vmax;
        out.total = total;
        reset();
        return out;
    }

    float avg() const {
        if (records.empty()) {
            return 0.f;
        }
        return static_cast<float>(total) / static_cast<float>(records.size());
    }

    T med() const {
        std::vector<T> cpy(records);
        std::nth_element(cpy.begin(), cpy.begin() + cpy.size() / 2, cpy.end());
        return cpy[cpy.size() / 2];
    }

    template <typename F>
    void record(F&& f) {
        static_assert(U::timing, "Only timing metrics can record a body");
        const auto start = std::chrono::steady_clock::now();
        f();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        *this += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000.0f;
    }

    void print(std::ostream& os, const std::string& label, int width = 44) const {
        format_guard fmt(os);

        const char* units = U::name;
        detail::print_label(os, label.empty() ? std::string("<unnamed timer>") : label, width);
        if constexpr (U::timing) {
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
        } else {
            os << "[ " << sum_f() << " " << units << " total in " << records.size() << " records ]";
        }
    }
};

template <class M>
struct Profile : public IProfile {
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
    // Set when sibling entries of the top level may genuinely overlap in time
    // (async funcall pipelining) - then no residual row is meaningful there.
    bool children_overlap = false;

    explicit Profile(Scope scope = Scope::Execution, bool on = enabled_by_default()) : m_on(on), m_scope(scope) {}
    Profile(const Profile&) = delete;
    Profile& operator=(const Profile&) = delete;

    bool enabled() const {
        return compiled_in && m_on;
    }

    // Set before first use; existing handles remain valid when disabling a profile.
    void enable(bool on) {
        m_on = on;
    }

    Scope scope() const override {
        return m_scope;
    }

    // Resolves a tag ONCE. Returns nullptr when the profile is disabled, so callers
    // can skip both the lookup and the timing. Mapped-value references of a std::map
    // are stable across inserts and across reset(), so the handle stays valid for
    // the profile's lifetime.
    M* handle(const std::string& tag) {
        if (!enabled()) {
            return nullptr;
        }
        return &ensure(tag);
    }

    template <typename F>
    M* handle_lazy(M*& cache, F&& make_tag) {
        if (!enabled()) {
            return nullptr;
        }
        if (cache == nullptr) {
            cache = &ensure(make_tag());
        }
        return cache;
    }

    void add(const std::string& tag, typename M::value_t value) {
        if (enabled()) {
            ensure(tag).add(value);
        }
    }

    // Gated stage timing for call sites that cannot pre-resolve a handle. When the
    // profile is disabled the body still runs and no map lookup happens. A tag that
    // the caller builds (not a literal) is still built.
    template <typename Tag, typename F>
    void record(const Tag& tag, F&& f) {
        if (!enabled()) {
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

    ~Profile() {
        if (enabled()) {
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
    bool m_on;
    Scope m_scope;
#ifndef NDEBUG
    // I1: a profile is only ever written by the thread that executes the corresponding
    // subgraph. Caught here rather than by silent map corruption later.
    mutable std::thread::id m_owner;
#endif

    void check_owner() const {
#ifndef NDEBUG
        if (m_scope != Scope::Execution || !enabled()) {
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
            auto [new_iter, unused] = metrics.emplace(tag, M{});
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
                detail::print_label(os, "(overlapped)", 44 - indent * 2);
                os << "[ children run concurrently - residual is not meaningful ]" << std::endl;
                return;
            }
            const float residual = rows[i].total - child_sum[i];
            const float tol = std::max(0.05f, 0.001f * std::fabs(rows[i].total));
            if (residual > tol) {
                os << pad(indent);
                detail::print_label(os, "(unattributed)", 44 - indent * 2);
                os << "[ total = " << residual << " ms ]" << std::endl;
            } else if (residual < -tol) {
                os << pad(indent);
                detail::print_label(os, "(overlapped)", 44 - indent * 2);
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

using ms_metric = metric<MSec>;
using bytes_metric = metric<Bytes>;
using metric_ptr = ms_metric*;

// Allocation-free RAII timer. A null handle makes it inert: no clock is read at all,
// which is the one predictable branch the disabled path is allowed to pay. In a
// non-debug-caps build it is always inert and the branch folds away.
class sample {
public:
    explicit sample(ms_metric* m) noexcept
        : m_metric(compiled_in ? m : nullptr),
          m_start(active() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}

    sample(const sample&) = delete;
    sample& operator=(const sample&) = delete;
    sample(sample&& other) noexcept
        : m_metric(std::exchange(other.m_metric, nullptr)),
          m_start(other.m_start),
          m_elapsed(other.m_elapsed),
          m_stopped(other.m_stopped),
          m_commit(std::exchange(other.m_commit, false)) {}
    sample& operator=(sample&&) = delete;

    float elapsed_ms() const noexcept {
        if (!active()) {
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
        if (!active()) {
            return 0.f;
        }
        if (!m_stopped) {
            m_elapsed = elapsed_ms();
            m_stopped = true;
        }
        m_commit = true;
        return m_elapsed;
    }

    float commit_at(std::chrono::steady_clock::time_point end) noexcept {
        if (!active()) {
            return 0.f;
        }
        // A missing or invalid completion timestamp must not produce a negative
        // or epoch-sized duration.
        const auto now = std::chrono::steady_clock::now();
        if (end < m_start || end > now) {
            end = now;
        }
        m_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count() / 1000.0f;
        m_stopped = true;
        m_commit = true;
        return m_elapsed;
    }

    void discard() noexcept {
        m_commit = false;
    }

    ~sample() noexcept {
        if (m_commit && active()) {
            // Recording allocates; a failure here must not turn optional profiling
            // into a std::terminate out of a noexcept destructor.
            try {
                *m_metric += m_elapsed;
            } catch (...) {
            }
        }
    }

private:
    bool active() const noexcept {
        return compiled_in && m_metric != nullptr;
    }

    ms_metric* m_metric = nullptr;
    std::chrono::steady_clock::time_point m_start;
    float m_elapsed = 0.f;
    bool m_stopped = false;
    bool m_commit = false;
};

// Request-local profiles and handles. Slot ids are real (prototype) submodel ids.
struct RequestPerf {
    struct SubmodelSlot {
        std::string tag;
        metric_ptr self = nullptr;
        metric_ptr device = nullptr;
        metric_ptr attributed = nullptr;
    };

    Profile<ms_metric> execution{Scope::Execution};
    Profile<bytes_metric> footprint{Scope::Lifetime};
    std::vector<SubmodelSlot> slots;

    explicit RequestPerf(std::size_t count) : slots(count) {}

    void begin_run() {
        if (!execution.enabled()) {
            return;
        }
        for (auto& slot : slots) {
            slot.attributed = nullptr;
        }
    }

    template <typename F>
    metric_ptr handle(metric_ptr& cache, F&& make_tag) {
        return execution.handle_lazy(cache, std::forward<F>(make_tag));
    }

    void attribute(std::size_t real_idx, metric_ptr handle) {
        if (execution.enabled()) {
            slots.at(real_idx).attributed = handle;
        }
    }

    template <typename F>
    void add_submodel(std::size_t real_idx, float elapsed, F&& make_tag) {
        if (!execution.enabled()) {
            return;
        }
        auto& slot = slots.at(real_idx);
        handle(slot.self, std::forward<F>(make_tag))->add(elapsed);
        if (slot.attributed != nullptr) {
            slot.attributed->add(elapsed);
        }
    }

    template <typename F>
    void add_footprint(const std::string& device, F&& bytes) {
        if (footprint.enabled()) {
            footprint.add(device, bytes());
        }
    }
};

}  // namespace perf
}  // namespace npuw
}  // namespace ov
