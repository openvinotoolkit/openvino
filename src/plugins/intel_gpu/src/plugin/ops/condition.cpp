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

static cldnn::topology::ptr gen_topology(Program& p, const std::shared_ptr<ngraph::Function> net) {
    InferenceEngine::CNNNetwork body_network(net);
    Program body_program(body_network, p.get_engine(), p.get_config(), true);
    return body_program.GetTopology();
}

// TODO: [If_op] Convert Model to primitive
// TODO: [If_op] Rename condition to If
// TODO: [If_op] Modify params by params of ngraph::if (cond{boolean})
static void CreateIfOp(Program& p, const std::shared_ptr<ngraph::op::v8::If>& op) {
    const std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    std::cout << "inputs : " << inputs.size() << std::endl;
    for (auto& in : inputs) {
        std::cout << "* " << in.idx << ", " << in.pid << std::endl;
    }

    auto& then_desc_vec = op->get_input_descriptions(0);
    {
        std::cout << "then_desc_vec=[";
        for (auto& in_desc : then_desc_vec) {
            std::cout << "{" << in_desc->m_body_parameter_index;
            std::cout << "," << in_desc->m_input_index << "},";
        }

        std::cout << "]" << std::endl;
    }
    auto& else_desc_vec = op->get_input_descriptions(1);
    {
        std::cout << "else_desc_vec=[{m_body_parameter_index, m_input_index}, ";
        for (auto& in_desc : else_desc_vec) {
            std::cout << "{" << in_desc->m_body_parameter_index;
            std::cout << "," << in_desc->m_input_index << "},";
        }

        std::cout << "]" << std::endl;
    }


    // op->get_input_size();
    cldnn::input_info input;

    auto topology_true  = *gen_topology(p, op->get_then_body());
    auto topology_false = *gen_topology(p, op->get_else_body());
    cldnn::primitive_id compare_data;
    cldnn::cond_functions func;

    const cldnn::condition conditionPrimitive(layerName,
                                {input},
                                topology_true,
                                topology_false,
                                compare_data,
                                func);

    p.add_primitive(*op, conditionPrimitive);
}

REGISTER_FACTORY_IMPL(v8, If);


}  // namespace intel_gpu
}  // namespace ov
