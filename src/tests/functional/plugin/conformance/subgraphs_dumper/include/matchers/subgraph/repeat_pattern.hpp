// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include "matchers/subgraph/subgraph.hpp"
#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class RepeatPatternExtractor final : public SubgraphExtractor {
public:
    RepeatPatternExtractor() {
        MatchersManager::MatchersMap matchers = {
            { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
            { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
        };
        manager.set_matchers(matchers);
    }

    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                        bool is_extract_body = true,
                                        bool is_copy_constants = true) override;

private:
    MatchersManager manager;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
