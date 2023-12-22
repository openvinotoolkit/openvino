// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph {
namespace op {
namespace util {
struct NGRAPH_API_DEPRECATED oi_pair {
    size_t output;
    size_t input;
    bool destructive;
};

/// \brief Base class for annotations added to graph ops
class NGRAPH_API_DEPRECATED NGRAPH_API OpAnnotations {
    NGRAPH_SUPPRESS_DEPRECATED_START
public:
    virtual ~OpAnnotations() = default;

    void add_in_place_oi_pair(const struct oi_pair& oi) {
        for (const auto& e : m_in_place_oi_pairs) {
            if (e.input == oi.input || e.output == oi.output) {
                OPENVINO_THROW("In_place hint conflicts with an existing entry");
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
