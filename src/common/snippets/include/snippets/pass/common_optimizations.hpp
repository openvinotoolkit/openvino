// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

class CommonOptimizations : public ov::pass::MatcherPass {
    class SubgraphPass;
    class SubgraphManager;
    friend class ExtractUnsupportedTransposes;
    friend class SplitDimensionM;

public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::CommonOptimizations");

    /**
     * @interface Config
     * @brief Configuration for CommonOptimizations pass
     * @ingroup snippets
     */
    struct Config {
        Config(size_t concurrency, bool split_m_dimension)
            : m_concurrency(concurrency),
              m_split_m_dimension(split_m_dimension) {
            OPENVINO_ASSERT(concurrency > 0, "Concurrency should be greater than 0");
        }

        [[nodiscard]] size_t get_concurrency() const {
            return m_concurrency;
        }

        [[nodiscard]] bool get_split_m_dimension() const {
            return m_split_m_dimension;
        }

    private:
        size_t m_concurrency = 0;
        // True if "SplitDimensionM" optimization is enabled.
        bool m_split_m_dimension = true;
    };

    explicit CommonOptimizations(const Config& config);
};

}  // namespace ov::snippets::pass
