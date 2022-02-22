// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/base_matcher.hpp"
#include "common_test_utils/common_utils.hpp"


SubgraphsDumper::iMatcherConfig::Ptr SubgraphsDumper::Matcher::get_config(const std::shared_ptr<ov::Node> &node) const {
    for (const auto &cfg : default_configs) {
        if (cfg->op_in_config(node)) {
            return cfg;
        }
    }
    for (const auto &cfg : default_configs) {
        if (cfg->is_fallback_config) {
            return cfg;
        }
    }
    return std::make_shared<MatcherConfig<>>();
}
