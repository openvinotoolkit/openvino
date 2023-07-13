// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class RepeatPatternMatcher : public SubgraphMatcher {
public:
    RepeatPatternMatcher() {
        MatchersManager::MatchersMap matchers = {
            { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
            { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
        };
        manager.set_matchers(matchers);
    }

    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;

private:
    MatchersManager manager;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
