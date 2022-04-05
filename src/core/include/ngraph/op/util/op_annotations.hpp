// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph {
namespace op {
namespace util {
struct NGRAPH_DEPRECATED("It is obsolete structure and will be removed soon") oi_pair {
    size_t output;
    size_t input;
    bool destructive;
};

/// \brief Base class for annotations added to graph ops

class NGRAPH_DEPRECATED("It is obsolete structure and will be removed soon") NGRAPH_API OpAnnotations {
    NGRAPH_SUPPRESS_DEPRECATED_START
public:
    virtual ~OpAnnotations() = default;

    void add_in_place_oi_pair(const struct oi_pair& oi) {
        for (auto e : m_in_place_oi_pairs) {
            if (e.input == oi.input || e.output == oi.output) {
                throw ngraph_error("In_place hint conflicts with an existing entry");
            }
        }
        m_in_place_oi_pairs.emplace_back(oi);
    }

    const std::vector<struct oi_pair>& get_in_place_oi_pairs() const {
        return m_in_place_oi_pairs;
    }
    bool is_cacheable() const {
        return m_cacheable;
    }
    void set_cacheable(bool val) {
        m_cacheable = val;
    }

private:
    // map of output-input pairs for which in-place computation is valid
    std::vector<struct oi_pair> m_in_place_oi_pairs;

    bool m_cacheable = false;
    NGRAPH_SUPPRESS_DEPRECATED_END
};
}  // namespace util
}  // namespace op
}  // namespace ngraph
