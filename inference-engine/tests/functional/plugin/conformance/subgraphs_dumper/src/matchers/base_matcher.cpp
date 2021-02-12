// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/base_matcher.hpp"
#include "common_test_utils/common_utils.hpp"

void SubgraphsDumper::Matcher::validate_and_unwrap_config() {
    std::map<std::string, size_t> ops_configs_counter;
    for (const auto &cfg : default_configs) {
        if (cfg.target_ops.empty()) {
            matcher_configs_unwraped["default"] = cfg;
        }
        for (const auto &op : cfg.target_ops) {
            if (ops_configs_counter.find(op) == ops_configs_counter.end()) {
                ops_configs_counter[op] = 1;
            } else {
                ops_configs_counter[op] += 1;
            }
            matcher_configs_unwraped[op] = cfg;
        }
    }
    bool has_config_duplicates = false;
    std::vector<std::string> ops_with_duplicates = {};
    // TODO: default configs duplication unvalidated
    for (const auto &it : ops_configs_counter) {
        if (it.second != 1) {
            has_config_duplicates = true;
            ops_with_duplicates.push_back(it.first);
        }
    }
    if (has_config_duplicates) {
        std::ostringstream msg;
        msg << "Multiple configurations for operations "
            << CommonTestUtils::vec2str(ops_with_duplicates) << " provided in matcher " << get_name();
        throw std::runtime_error(msg.str());
    }
}
