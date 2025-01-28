// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_load_store_and_convert.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"

bool ov::intel_cpu::pass::FuseLoadStoreConvert::fuse_load_convert(
    snippets::lowered::LinearIR& linear_ir,
    snippets::lowered::LinearIR::constExprIt& convert_it) {
    const auto& convert_expr = *convert_it;
    const auto& convert = ov::as_type_ptr<ov::op::v0::Convert>(convert_expr->get_node());
    const auto& input_connector = convert_expr->get_input_port_connector(0);
    if (convert->get_destination_type() != ov::element::f32 && convert->get_destination_type() != ov::element::i32) {
        return false;
    }

    const auto& load_output = input_connector->get_source();
    const auto& load_expr = load_output.get_expr();
    const auto load = ov::as_type_ptr<snippets::op::Load>(load_expr->get_node());
    if (!load || ov::is_type<snippets::op::LoadReorder>(load_expr->get_node()) ||
        ov::is_type<snippets::op::BroadcastLoad>(load_expr->get_node())) {
        return false;
    }

    const auto consumers = input_connector->get_consumers();
    if (consumers.size() != 1) {
        return false;
    }

    OPENVINO_ASSERT(convert_expr->get_loop_ids() == load_expr->get_loop_ids(),
                    "The pair of Load and Convert expressions must be in the same loops!");

    const auto& parent_source = load_expr->get_input_port_connector(0)->get_source();
    const auto parent_output = parent_source.get_expr()->get_node()->output(parent_source.get_index());
    std::shared_ptr<ov::Node> load_convert = nullptr;
    if (ov::is_type<snippets::op::ConvertSaturation>(convert)) {
        load_convert = std::make_shared<ov::intel_cpu::LoadConvertSaturation>(parent_output,
                                                                              convert->get_destination_type(),
                                                                              load->get_count(),
                                                                              load->get_offset());
    } else if (ov::is_type<snippets::op::ConvertTruncation>(convert)) {
        load_convert = std::make_shared<ov::intel_cpu::LoadConvertTruncation>(parent_output,
                                                                              convert->get_destination_type(),
                                                                              load->get_count(),
                                                                              load->get_offset());
    } else {
        OPENVINO_THROW("Type of Convert op is undefined. Supports only fusing Load and ConvertTruncation or "
                       "ConvertSaturation ops");
    }

    convert_it = linear_ir.replace_with_node({load_expr, convert_expr}, load_convert);

    return true;
}

bool ov::intel_cpu::pass::FuseLoadStoreConvert::fuse_store_convert(
    snippets::lowered::LinearIR& linear_ir,
    snippets::lowered::LinearIR::constExprIt& convert_it) {
    const auto& convert_expr = *convert_it;
    const auto& convert = ov::as_type_ptr<ov::op::v0::Convert>(convert_expr->get_node());
    const auto& output_connector = convert_expr->get_output_port_connector(0);
    if (convert->get_input_element_type(0) != ov::element::f32 &&
        convert->get_input_element_type(0) != ov::element::i32) {
        return false;
    }

    const auto consumers = output_connector->get_consumers();
    if (consumers.size() != 1) {
        return false;
    }

    const auto store_input = *(consumers.begin());
    const auto& store_expr = store_input.get_expr();
    const auto store = ov::as_type_ptr<snippets::op::Store>(store_expr->get_node());
    if (!store) {
        return false;
    }

    OPENVINO_ASSERT(convert_expr->get_loop_ids() == store_expr->get_loop_ids(),
                    "The pair of Convert and Store expressions must be in the same loops!");

    const auto& parent_source = convert_expr->get_input_port_connector(0)->get_source();
    const auto parent_output = parent_source.get_expr()->get_node()->output(parent_source.get_index());
    std::shared_ptr<ov::Node> store_convert = nullptr;
    if (ov::is_type<snippets::op::ConvertSaturation>(convert)) {
        store_convert = std::make_shared<ov::intel_cpu::StoreConvertSaturation>(parent_output,
                                                                                convert->get_destination_type(),
                                                                                store->get_count(),
                                                                                store->get_offset());
    } else if (ov::is_type<snippets::op::ConvertTruncation>(convert)) {
        store_convert = std::make_shared<ov::intel_cpu::StoreConvertTruncation>(parent_output,
                                                                                convert->get_destination_type(),
                                                                                store->get_count(),
                                                                                store->get_offset());
    } else {
        OPENVINO_THROW("Type of Convert op is undefined. Supports only fusing Store and ConvertTruncation or "
                       "ConvertSaturation ops");
    }

    convert_it = linear_ir.replace_with_node({convert_expr, store_expr}, store_convert);

    return true;
}

bool ov::intel_cpu::pass::FuseLoadStoreConvert::run(snippets::lowered::LinearIR& linear_ir,
                                                    snippets::lowered::LinearIR::constExprIt begin,
                                                    snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::FuseLoadStoreConvert")

    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& convert = expr->get_node();
        if (!ov::is_type<ov::op::v0::Convert>(convert)) {
            continue;
        }

        if (fuse_load_convert(linear_ir, expr_it)) {
            modified = true;
            continue;
        }
        if (fuse_store_convert(linear_ir, expr_it)) {
            modified = true;
            continue;
        }
    }

    return modified;
}
