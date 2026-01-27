// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

class CommonOptimizations : public ov::pass::MatcherPass {
    class SubgraphPass;
    class SubgraphManager;
    friend class ExtractConstants;
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
        using TransposeSupportCallback = std::function<bool(const std::shared_ptr<const ov::Node>&)>;

        Config(size_t concurrency, bool split_m_dimension)
            : m_concurrency(concurrency),
              m_split_m_dimension(split_m_dimension) {
            OPENVINO_ASSERT(concurrency > 0, "Concurrency should be greater than 0");
        }

        void set_concurrency(size_t concurrency) {
            OPENVINO_ASSERT(concurrency > 0, "Concurrency should be greater than 0");
            m_concurrency = concurrency;
        }

        [[nodiscard]] size_t get_concurrency() const {
            return m_concurrency;
        }

        [[nodiscard]] bool get_split_m_dimension() const {
            return m_split_m_dimension;
        }

        void set_transpose_support_callback(TransposeSupportCallback cb) {
            m_transpose_support_cb = std::move(cb);
        }

        [[nodiscard]] const TransposeSupportCallback& get_transpose_support_callback() const {
            return m_transpose_support_cb;
        }

    private:
        size_t m_concurrency = 0;
        // True if "SplitDimensionM" optimization is enabled.
        bool m_split_m_dimension = true;
        // Callback to determine whether a given Transpose is supported inside Subgraph.
        // If empty, all Transposes are treated as unsupported.
        TransposeSupportCallback m_transpose_support_cb = [](const std::shared_ptr<const ov::Node>&) {
            return false;
        };
    };

    explicit CommonOptimizations(const Config& config);
};

}  // namespace ov::snippets::pass
