// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/serialize_control_flow.hpp"

#include "openvino/pass/serialize.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/serialization_node.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SerializeControlFlow::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SerializeControlFlow")
    if (linear_ir.empty())
        return false;

    auto first_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    first_node->set_friendly_name("Start");
    first_node->get_rt_info()["execTimeMcs"] = 0;
    std::shared_ptr<Node> serialization_node = first_node;

    // This map allows to get LoopBegin serialization node by original LoopBegin node
    // It is used to draw an edge between LoopBegin and LoopEnd serialization nodes
    std::map<std::shared_ptr<snippets::op::LoopBegin>, std::shared_ptr<Node>> loops_map;
    for (const auto& expr : linear_ir) {
        const auto node = expr->get_node();
        if (auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(node)) {
            OPENVINO_ASSERT(loops_map.count(loop_end->get_loop_begin()),
                            "Serialization can't find LoopBegin that corresponds to LoopEnd with friendly name ",
                            loop_end->get_friendly_name());
            auto loop_begin_serialization_node = loops_map.at(loop_end->get_loop_begin());
            serialization_node = std::make_shared<op::SerializationNode>(ov::OutputVector{serialization_node, loop_begin_serialization_node}, expr);
        } else {
            serialization_node = std::make_shared<op::SerializationNode>(ov::OutputVector{serialization_node}, expr);
            if (auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(node)) {
                loops_map[loop_begin] = serialization_node;
            }
        }
    }
    auto last_node = std::make_shared<ov::op::v0::Result>(serialization_node);
    last_node->set_friendly_name("End");
    const auto model = std::make_shared<ov::Model>(ResultVector{last_node}, ParameterVector{first_node}, "Lowered_IR_Control_Flow");
    return ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
