// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "form_parallel_subgraphs.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/subgraph.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

FormParallelSubgraphs::FormParallelSubgraphs() {
    MATCHER_SCOPE(FormParallelSubgraphs);

    auto parallel_candidate = [](ov::Output<ov::Node> output) -> bool {
        const auto& node = *output.get_node();
        return node.get_rt_info().count("parallelDomain");
    };

    auto candidate_m = ov::pass::pattern::any_input(parallel_candidate);

    ov::matcher_pass_callback callback = [candidate_m](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& candidate_out = pattern_map.at(candidate_m);
        const auto& candidate = candidate_out.get_node_shared_ptr();

        if (candidate->get_type_info() == SubModel::get_type_info_static()) {
            return false;  // avoid recursive wrapping submodel into submodel
        }

        ov::ParameterVector params;
        ov::ResultVector results;
        OutputVector args;

        ov::OutputVector candidate_inputs;
        for (size_t i = 0; i < candidate->inputs().size(); i++) {
            const auto& input = candidate->get_input_node_shared_ptr(i);
            if (!op::util::is_on_constant_path(input->output(0))) {
                params.emplace_back(std::make_shared<ov::op::v0::Parameter>(candidate->get_element_type(),
                                                                            candidate->get_input_partial_shape(0)));
                candidate_inputs.push_back(params.back()->output(0));
            } else {
                candidate_inputs.emplace_back(candidate->get_input_source_output(i));
            }
        }

        auto candidate_clone = candidate->clone_with_new_inputs(candidate_inputs);

        for (const auto& output : candidate_clone->outputs()) {
            results.emplace_back(std::make_shared<ov::op::v0::Result>(output));
        }

        auto submodel =
            std::make_shared<ov::Model>(results, params, candidate_clone->get_friendly_name() + "_subgraph");
        auto subgraph = std::make_shared<ov::intel_cpu::SubModel>(submodel);
        subgraph->set_friendly_name("Submodel_" + candidate_clone->get_friendly_name());

        for (size_t i = 0; i < params.size(); i++) {
            subgraph->set_invariant_input(params[i], candidate->input_value(i));
        }

        ov::copy_runtime_info(subgraph, candidate);
        ov::replace_node_update_name(candidate, subgraph);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(candidate_m, matcher_name);
    this->register_matcher(m, callback);
}
}  // namespace intel_cpu
}  // namespace ov
