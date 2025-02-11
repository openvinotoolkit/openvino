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

namespace ov {
namespace npuw {
namespace perf {

float ms_to_run(std::function<void()>&& body);

struct MSec {
    constexpr static const char* name = "ms";
};

template <typename T, typename U>
class metric {
    std::vector<T> records;
    T vmin = std::numeric_limits<T>::max();
    T vmax = std::numeric_limits<T>::min();
    std::string unit;

public:
    void operator+=(T&& t) {
        vmin = std::min(vmin, t);
        vmax = std::max(vmax, t);
        records.push_back(std::move(t));
    }
    float avg() const {
        float acc = 0.f;
        for (auto&& t : records) {
            acc += static_cast<float>(t) / records.size();
        }
        return acc;
    }
    T med() const {
        std::vector<T> cpy(records);
        std::nth_element(cpy.begin(), cpy.begin() + cpy.size() / 2, cpy.end());
        return cpy[cpy.size() / 2];
    }
    friend std::ostream& operator<<(std::ostream& os, const metric<T, U>& m) {
        const char* units = U::name;
        os << "[ min = " << m.vmin << units << ", avg = " << m.avg() << units << ", med = " << m.med() << units
           << ", max = " << m.vmax << units << " in " << m.records.size() << " records ]";
        return os;
    }
};

}  // namespace perf
}  // namespace npuw
}  // namespace ov
