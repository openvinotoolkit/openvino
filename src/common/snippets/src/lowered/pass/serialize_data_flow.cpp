// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/serialize_data_flow.hpp"

#include "openvino/pass/serialize.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/serialization_node.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SerializeDataFlow::run(const LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SerializeDataFlow")
    if (linear_ir.empty())
        return false;

    ov::ResultVector results;
    ov::ParameterVector parameters;
    std::map<ExpressionPtr, std::shared_ptr<Node>> ops_map;
    const auto serialization_mode = op::SerializationNode::SerializationMode::DATA_FLOW;
    for (const auto& expr : linear_ir) {
        const auto node = expr->get_node();
        ov::OutputVector inputs(expr->get_input_count());
        for (size_t i = 0; i < expr->get_input_count(); ++i) {
            const auto& input_expr = expr->get_input_port_connector(i)->get_source().get_expr();
            OPENVINO_ASSERT(ops_map.count(input_expr), "input node wasn't found during serialization");
            inputs[i] = ops_map[input_expr]->output(expr->get_input_port_connector(i)->get_source().get_index());
        }
        if (ov::is_type<ov::op::v0::Parameter>(node)) {
            const auto parameter = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
            ops_map[expr] = parameter;
            parameters.push_back(parameter);
        } else if (ov::is_type<ov::op::v0::Result>(node)) {
            const auto result = std::make_shared<ov::op::v0::Result>(inputs[0]);
            ops_map[expr] = result;
            results.push_back(result);
        } else {
            const auto serialization_node = std::make_shared<op::SerializationNode>(inputs, expr, serialization_mode);
            ops_map[expr] = serialization_node;
        }
    }
    const auto model = std::make_shared<ov::Model>(results, parameters, "Lowered_IR_Data_Flow");
    return ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
