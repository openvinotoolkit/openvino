// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"

#include <cpp/ie_cnn_network.h>

#include "ngraph/op/if.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "transformations/utils/utils.hpp"
#include "ie_ngraph_utils.hpp"

#include "intel_gpu/primitives/condition.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/graph/topology.hpp"

#include <vector>
#include <algorithm>

namespace ov {
namespace intel_gpu {

const size_t idx_true = 0;
const size_t idx_false = 1;

static cldnn::topology::ptr gen_topology(Program& p, const std::shared_ptr<ngraph::Function> net) {
    InferenceEngine::CNNNetwork body_network(net);
    Program body_program(body_network, p.get_engine(), p.get_config(), true);
    return body_program.GetTopology();
}

static cldnn::condition::branch_info gen_branch_info(Program& p, const std::shared_ptr<ngraph::op::v8::If>& op, size_t idx) {
    cldnn::condition::branch_info branch;
    const auto& internal_body = (idx == idx_true)? op->get_then_body() : op->get_else_body();

    branch.topology_ptr = gen_topology(p, internal_body);

    auto& input_map = branch.input_map;
    auto external_inputs = p.GetInputInfo(op);
    auto internal_inputs = internal_body->get_parameters();
    auto input_desc_vec = op->get_input_descriptions(idx);
    for (auto& in_desc : input_desc_vec) {
        const auto& external_id = external_inputs.at(in_desc->m_input_index).pid;
        const auto& internal_id = layer_type_name_ID(internal_inputs.at(in_desc->m_body_parameter_index));
        input_map.insert({external_id, internal_id});
    }

    auto& output_map = branch.output_map;
    auto internal_outputs = internal_body->get_results();
    // std::cout << "inner body outputs: " << internal_outputs.size() << std::endl;
    auto output_desc_vec = op->get_output_descriptions(idx);
    for (auto& out_desc : output_desc_vec) {
        const auto& internal_id = layer_type_name_ID(internal_outputs.at(out_desc->m_body_value_index));
        output_map.insert({out_desc->m_output_index, internal_id});
    }

    return branch;
}

static void CreateIfOp(Program& p, const std::shared_ptr<ngraph::op::v8::If>& op) {
    auto inputs = p.GetInputInfo(op);
    OPENVINO_ASSERT(inputs.size() >= 1, "Invalid inputs count (Not allowed no input)");

    const std::string layerName = layer_type_name_ID(op);
    auto branch_true = gen_branch_info(p, op, idx_true);
    auto branch_false = gen_branch_info(p, op, idx_false);

    const cldnn::condition conditionPrimitive(layerName,
                                inputs,
                                branch_true,
                                branch_false);

    p.add_primitive(*op, conditionPrimitive);
}

REGISTER_FACTORY_IMPL(v8, If);


}  // namespace intel_gpu
}  // namespace ov
