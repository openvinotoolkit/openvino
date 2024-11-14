// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::FoldSubgraphEmptyInputs::FoldSubgraphEmptyInputs() {
    MATCHER_SCOPE(FoldSubgraphEmptyInputs);
    auto multi_subgraph_op_pattern = pattern::wrap_type<op::util::MultiSubGraphOp>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(m.get_match_root());
        if (multi_subgraph_op == nullptr) {
            return false;
        }
        const auto& rt_info = multi_subgraph_op->get_rt_info();
        if (rt_info.count(DisableFoldSubgraphEmptyInputs::get_type_info_static())) {
            return false;
        }
        auto multi_subgraph_op_inputs = multi_subgraph_op->input_values();

        std::vector<ov::Output<ov::Node>> empty_inputs;
        std::copy_if(std::begin(multi_subgraph_op_inputs),
                     std::end(multi_subgraph_op_inputs),
                     std::back_inserter(empty_inputs),
                     [](const Output<Node>& input) {
                         // skip constants
                         if (ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr())) {
                             return false;
                         }
                         // skip non-static shapes
                         const auto& in_shape = input.get_partial_shape();
                         if (in_shape.is_dynamic()) {
                             return false;
                         }
                         return std::any_of(std::begin(in_shape), std::end(in_shape), [](const ov::Dimension& dim) {
                             return dim.get_length() == 0;
                         });
                     });

        if (empty_inputs.size()) {
            for (const auto& input : empty_inputs) {
                const ov::Output<ov::Node> const_empty_replacement =
                    std::make_shared<ov::op::v0::Constant>(input.get_element_type(), input.get_shape());
                std::replace(std::begin(multi_subgraph_op_inputs),
                             std::end(multi_subgraph_op_inputs),
                             input,
                             const_empty_replacement);
                copy_runtime_info(input.get_node_shared_ptr(), const_empty_replacement.get_node_shared_ptr());
            }
            multi_subgraph_op->set_arguments(multi_subgraph_op_inputs);
            return true;
        }
        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(multi_subgraph_op_pattern, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::disable_fold_subgraph_empty_inputs(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info().emplace(DisableFoldSubgraphEmptyInputs::get_type_info_static(),
                                DisableFoldSubgraphEmptyInputs{});
}

void ov::pass::enable_fold_subgraph_empty_inputs(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info().erase(DisableFoldSubgraphEmptyInputs::get_type_info_static());
}

bool ov::pass::fold_subgraph_empty_inputs_is_disabled(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().count(DisableFoldSubgraphEmptyInputs::get_type_info_static());
}
