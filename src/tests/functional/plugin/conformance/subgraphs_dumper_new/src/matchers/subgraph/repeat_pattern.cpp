// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <vector>

#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"

#include "matchers/subgraph/repeat_pattern.hpp"

#include "openvino/op/util/op_types.hpp"
#include "matchers/manager.hpp"

#include "utils/node.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

std::list<ExtractedPattern>
RepeatPatternMatcher::extract(const std::shared_ptr<ov::Model> &model) {
    std::unordered_set<std::string> checked_ops;
    auto ordered_ops = model->get_ordered_ops();
    std::list<ExtractedPattern> to_cache;

    MatchersManager::MatchersMap matchers = {
        { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
        { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
    };
    MatchersManager manager;
    manager.set_matchers(matchers);

    for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
        auto op = ordered_ops[idx];
        auto op_name = op->get_friendly_name();
        if (checked_ops.count(op_name)|| ov::op::util::is_constant(op) || ov::op::util::is_parameter(op) || ov::op::util::is_output(op)) {
            continue;
        }

        std::vector<size_t> tmp_buf;
        for (size_t i = idx; i < ordered_ops.size(); ++i) {
            if (manager.match(op, ordered_ops[i])) {
                tmp_buf.push_back(i);
            }
        }
        if (tmp_buf.size() < 2) {
            checked_ops.insert(op->get_friendly_name());
            continue;
        }

        std::vector<std::set<std::shared_ptr<ov::Node>>> to_generate_model(tmp_buf.size());
        for (size_t i = 0; i < tmp_buf.size(); ++i) {
            for (size_t j = 1; j < tmp_buf.size(); ++j) {
                size_t node_idx = tmp_buf[i], ref_node_idx = tmp_buf[j], it = 0;
                while (node_idx + it < ordered_ops.size() && ref_node_idx + it < ordered_ops.size()) {
                    if (manager.match(ordered_ops[node_idx + it], ordered_ops[ref_node_idx + it])) {
                        to_generate_model[i].insert(ordered_ops[node_idx + it]);
                        to_generate_model[j].insert(ordered_ops[ref_node_idx + it]);
                    }
                }
            }
        }
        for (size_t i = 0; i < tmp_buf.size(); ++i) {
            to_cache.push_back(
                generate_model(to_generate_model[i], ordered_ops[tmp_buf[i]], checked_ops));
        }
    }
    return to_cache;
}


