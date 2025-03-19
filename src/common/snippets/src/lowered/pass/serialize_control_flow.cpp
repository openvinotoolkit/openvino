// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/serialize_control_flow.hpp"

#include "openvino/pass/serialize.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/serialization_node.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/linear_ir_builder.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SerializeControlFlow::run(const LinearIR& original_linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SerializeControlFlow")
    if (original_linear_ir.empty())
        return false;
    const auto& linear_ir = m_update_dynamic_ops ? LinearIRBuilder().clone(original_linear_ir) : original_linear_ir;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info_map = loop_manager ? loop_manager->get_map() : std::map<size_t, LoopInfoPtr>{};

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
            if (m_update_dynamic_ops) {
                OPENVINO_ASSERT(loop_info_map.count(loop_end->get_id()), "Failed to find loop id in loop info map");
                const auto& loop_info = loop_info_map.at(loop_end->get_id());
                loop_end->set_work_amount(loop_info->get_work_amount());
                loop_end->set_increment(loop_info->get_increment());
                loop_end->set_is_incremented(loop_info->get_is_incremented());
                if (auto unified = ov::as_type_ptr<UnifiedLoopInfo>(loop_info)) {
                    loop_end->set_ptr_increments(unified->get_ptr_increments());
                    loop_end->set_finalization_offsets(unified->get_finalization_offsets());
                } else if (auto expanded = ov::as_type_ptr<ExpandedLoopInfo>(loop_info)) {
                    loop_end->set_ptr_increments(expanded->get_ptr_increments());
                    loop_end->set_finalization_offsets(expanded->get_finalization_offsets());
                } else {
                    OPENVINO_THROW("Unknown LoopInfo type");
                }
            }
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
