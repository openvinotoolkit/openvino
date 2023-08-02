// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"
#include "matchers/single_op/manager.hpp"
#include "matchers/subgraph/subgraph.hpp"

using namespace ov::tools::subgraph_dumper;

bool
SubgraphExtractor::match(const std::shared_ptr<ov::Model> &model,
                         const std::shared_ptr<ov::Model> &ref_model) const {
    bool res = comparator.compare(model, ref_model).valid;
    if (res) {
        return res;
    }
    std::vector<std::shared_ptr<ov::Node>> ordered_ops = model->get_ordered_ops(),
                                           ref_ordered_ops = ref_model->get_ordered_ops();
    if (ordered_ops.size() != ref_ordered_ops.size())
        return false;

    MatchersManager::MatchersMap matchers = {
        { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
        { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
    };
    MatchersManager manager(matchers);
    for (size_t i = 0; i < ordered_ops.size(); ++i) {
        if (is_node_to_skip(ordered_ops[i]) && is_node_to_skip(ref_ordered_ops[i]))
            continue;
        if (!manager.match(ordered_ops[i], ref_ordered_ops[i])) {
            return false;
        }
    }
    return true;
}
