// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <ngraph/log.hpp>
#include <set>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ngraph {
using ov::graph_rewrite_callback;
using ov::handler_callback;
using ov::matcher_pass_callback;
using ov::recurrent_graph_rewrite_callback;
namespace pass {
using ov::pass::BackwardGraphRewrite;
using ov::pass::GraphRewrite;
using ov::pass::MatcherPass;

class NGRAPH_DEPRECATED("Use MatcherPass or FunctionPass instead.") NGRAPH_API RecurrentGraphRewrite
    : public FunctionPass {
public:
    RecurrentGraphRewrite(size_t num_iters = 10) : ModelPass(), m_num_iters(num_iters) {}

    void add_matcher(const std::shared_ptr<pattern::RecurrentMatcher>& m,
                     const ov::recurrent_graph_rewrite_callback& callback,
                     const PassPropertyMask& property) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        m_matchers.push_back(std::make_shared<MatcherPass>(
            "Recurrent matcher",
            nullptr,
            [m, callback](const std::shared_ptr<Node>& node) {
                NGRAPH_DEBUG << "Running recurrent matcher on " << node;
                if (m->match(node->output(0))) {
                    NGRAPH_DEBUG << "Recurrent matcher matched " << m.get();
                    return callback(*m.get());
                }
                return false;
            },
            property));
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    // TODO: This interface may deprecate after all passes are refactored.
    void add_matcher(const std::shared_ptr<pattern::RecurrentMatcher>& m,
                     const ov::recurrent_graph_rewrite_callback& callback) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        // TODO: before deprecate this function, by default expect the
        // callback require static shape.
        add_matcher(m, callback, {PassProperty::REQUIRE_STATIC_SHAPE});
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override {
        NGRAPH_SUPPRESS_DEPRECATED_START
        bool changed = false;
        size_t i = 0;

        auto run_matchers = [&]() -> bool {
            for (const auto& node : m->get_ops()) {
                for (auto& m_pass : m_matchers) {
                    if (m_pass->apply(node)) {
                        return true;
                    }
                }
            }
            return false;
        };

        do {
            changed = run_matchers();
            i++;
        } while (changed && i < m_num_iters);
        return changed;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

private:
    size_t m_num_iters;

    std::vector<std::shared_ptr<ov::pass::MatcherPass>> m_matchers;
};
}  // namespace pass
}  // namespace ngraph
