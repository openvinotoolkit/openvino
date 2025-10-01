// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "logging.hpp"

namespace ov {
namespace npuw {
namespace perf {

float ms_to_run(const std::function<void()>& body);

struct MSec {
    constexpr static const char* name = "ms";

    using type_t = float;

    static type_t sample(const std::function<void()>& body) {
        return ms_to_run(body);
    }
};

struct Bytes {
    constexpr static const char* name = "MB";
    constexpr static float scaler = 1024.0 * 1024;

    using type_t = uint64_t;
};

template <typename U>
class metric {
    using T = typename U::type_t;

private:
    std::vector<T> records;
    T vmin = std::numeric_limits<T>::max();
    T vmax = std::numeric_limits<T>::min();
    T total = 0;
    std::string name;
    bool enabled = false;

public:
    metric() = default;
    metric(metric&& m) : records(std::move(m.records)), name(std::move(m.name)), enabled(m.enabled) {}

    explicit metric(const std::string& named, bool active = false) : name(named), enabled(active) {}

    void enable() {
        enabled = true;
    }

    void operator+=(T&& t) {
        if (!enabled) {
            return;
        }
        vmin = std::min(vmin, t);
        vmax = std::max(vmax, t);
        records.push_back(std::move(t));
        total += t;
    }

    float avg() const {
        NPUW_ASSERT(enabled);
        float acc = 0.f;
        for (auto&& t : records) {
            acc += static_cast<float>(t) / records.size();
        }
        return acc;
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
            *this += U::sample(f);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const metric<U>& m) {
        const char* units = U::name;
        os << std::left << std::setw(20) << (m.name.empty() ? std::string("<unnamed timer>") : m.name);
        if (m.enabled) {
            if (m.records.empty()) {
                os << "[ no data ]";
            } else {
                os << "[ avg = " << m.avg() << " " << units << ", med = " << m.med() << " " << units << " in " << m.vmin
                   << ".." << m.vmax << " " << units << " range over " << m.records.size() << " records"
                   << ", total = " << m.total << units << " ]";
            }
        } else {
            os << "[ disabled ]";
        }
        return os;
    }
};

template <typename U>
class counter {
public:
    using T = typename U::type_t;

private:
    T total = 0;
    std::size_t calls = 0;
    std::string name;
    bool enabled = false;

public:
    counter() = default;
    counter(counter&& c) : total(std::move(c.total)), calls(c.calls), name(std::move(c.name)), enabled(c.enabled) {}

    explicit counter(const std::string& named, bool active = false) : name(named), enabled(active) {}

    void enable() {
        enabled = true;
    }

    void operator+=(T&& t) {
        if (enabled) {
            total += t;
            calls++;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const counter<U>& c) {
        const char* units = U::name;
        os << std::left << std::setw(20) << (c.name.empty() ? std::string("<unnamed counter>") : c.name);
        if (c.enabled) {
            os << "[ " << static_cast<decltype(U::scaler)>(c.total) / U::scaler << " " << units << " total in "
               << c.calls << " records ]";
        } else {
            os << "[ disabled ]";
        }
        return os;
    }
};

template <class Metric>
struct Profile {
    std::map<std::string, Metric> metrics;
    std::string area;
    bool report_on_die = false;

    Metric& operator[](const std::string& tag) {
        auto iter = metrics.find(tag);
        if (iter == metrics.end()) {
            // Use report_on_die as a "enabled" marker here
            auto [new_iter, unused] = metrics.insert({tag, Metric(tag, report_on_die)});
            return new_iter->second;
        }
        return iter->second;
    }

    void report() const {
        if (metrics.empty()) {
            return;
        }
        if (!area.empty()) {
            std::cout << area << ":" << std::endl;
        } else {
            std::cout << std::hex << this << ":" << std::endl;
        }
        for (auto&& m : metrics) {
            std::cout << "  " << m.second << std::endl;
        }
    }

    ~Profile() {
        if (report_on_die) {
            report();
        }
    }
};

}  // namespace perf
}  // namespace npuw
}  // namespace ov
