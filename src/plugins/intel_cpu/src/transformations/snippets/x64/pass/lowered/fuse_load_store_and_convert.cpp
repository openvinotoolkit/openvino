// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "fuse_load_store_and_convert.hpp"
#include "snippets/snippets_isa.hpp"

#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"


bool ov::intel_cpu::pass::FuseLoadStoreConvert::fuse_load_convert(ngraph::snippets::lowered::LinearIR& linear_ir,
                                                                  ngraph::snippets::lowered::LinearIR::constExprIt& convert_it) {
    const auto& convert_expr = *convert_it;
    const auto& convert = ov::as_type_ptr<ov::op::v0::Convert>(convert_expr->get_node());
    const auto& input_td = convert_expr->get_input_tensor(0);
    if (convert->get_destination_type() != ov::element::f32 && convert->get_destination_type() != ov::element::i32)
        return false;

    const auto& load_output = input_td->get_source();
    const auto& load_expr = load_output.get_expr();
    const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(load_expr->get_node());
    if (!load ||
        ov::is_type<ngraph::snippets::op::LoadReshape>(load_expr->get_node()) ||
        ov::is_type<ngraph::snippets::op::BroadcastLoad>(load_expr->get_node()))
        return false;

    const auto consumers = input_td->get_consumers();
    if (consumers.size() != 1)
        return false;

    std::shared_ptr<ngraph::Node> load_convert = nullptr;
    if (const auto convert_saturation = ov::as_type_ptr<ngraph::snippets::op::ConvertSaturation>(convert)) {
        load_convert = std::make_shared<ov::intel_cpu::LoadConvertSaturation>(load->input_value(0),
                                                                              convert_saturation->get_destination_type(),
                                                                              load->get_count(), load->get_offset());
    } else if (const auto convert_truncation = ov::as_type_ptr<ngraph::snippets::op::ConvertTruncation>(convert)) {
        load_convert = std::make_shared<ov::intel_cpu::LoadConvertTruncation>(load->input_value(0),
                                                                              convert_truncation->get_destination_type(),
                                                                              load->get_count(), load->get_offset());
    } else {
        OPENVINO_THROW("Type of Convert op is undefined. Supports only fusing Load and ConvertTruncation or ConvertSaturation ops");
    }

    const auto out_port = convert_expr->get_output_port(0);
    const auto convert_consumers = out_port.get_connected_ports();
    ngraph::snippets::lowered::PortManager::set_port_descriptor_ptr(load_convert->output(0), out_port.get_descriptor_ptr()->clone());
    const auto load_convert_expr = linear_ir.create_expression(load_convert, { load_expr->get_input_tensor(0) });
    const auto convert_expr_it = convert_it;
    const auto insertion_pos = std::next(convert_it);
    convert_it = linear_ir.insert(insertion_pos, load_convert_expr);
    linear_ir.erase(std::find(linear_ir.cbegin(), convert_expr_it, load_expr));
    linear_ir.erase(convert_expr_it);
    linear_ir.replace_input(convert_consumers, load_convert_expr->get_output_tensor(0));
    return true;
}

bool ov::intel_cpu::pass::FuseLoadStoreConvert::fuse_store_convert(ngraph::snippets::lowered::LinearIR& linear_ir,
                                                                   ngraph::snippets::lowered::LinearIR::constExprIt& convert_it) {
    const auto& convert_expr = *convert_it;
    const auto& convert = convert_expr->get_node();
    const auto& input_td = convert_expr->get_input_tensor(0);
    const auto& output_td = convert_expr->get_output_tensor(0);
    if (convert->get_input_element_type(0) != ov::element::f32 && convert->get_input_element_type(0) != ov::element::i32)
        return false;

    const auto consumers = output_td->get_consumers();
    if (consumers.size() != 1)
        return false;

    const auto store_input = *(consumers.begin());
    const auto& store_expr = store_input.get_expr();
    const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(store_expr->get_node());
    if (!store)
        return false;

    std::shared_ptr<ngraph::Node> store_convert = nullptr;
    if (const auto convert_saturation = ov::as_type_ptr<ngraph::snippets::op::ConvertSaturation>(convert)) {
        store_convert = std::make_shared<ov::intel_cpu::StoreConvertSaturation>(convert->input_value(0),
                                                                                convert_saturation->get_destination_type(),
                                                                                store->get_count(), store->get_offset());
    } else if (const auto convert_truncation = ov::as_type_ptr<ngraph::snippets::op::ConvertTruncation>(convert)) {
        store_convert = std::make_shared<ov::intel_cpu::StoreConvertTruncation>(convert->input_value(0),
                                                                                convert_truncation->get_destination_type(),
                                                                                store->get_count(), store->get_offset());
    } else {
        OPENVINO_THROW("Type of Convert op is undefined. Supports only fusing Store and ConvertTruncation or ConvertSaturation ops");
    }

    const auto out_port = store_expr->get_output_port(0);
    const auto store_consumers = out_port.get_connected_ports();
    ngraph::snippets::lowered::PortManager::set_port_descriptor_ptr(store_convert->output(0), out_port.get_descriptor_ptr()->clone());
    const auto store_convert_expr = linear_ir.create_expression(store_convert, { input_td });
    const auto convert_expr_it = convert_it;
    const auto insertion_pos = std::next(convert_it);
    convert_it = linear_ir.insert(insertion_pos, store_convert_expr);
    linear_ir.erase(std::find(convert_expr_it, linear_ir.cend(), store_expr));
    linear_ir.erase(convert_expr_it);
    linear_ir.replace_input(store_consumers, store_convert_expr->get_output_tensor(0));
    return true;
}

bool ov::intel_cpu::pass::FuseLoadStoreConvert::run(ngraph::snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::FuseLoadStoreConvert")

    bool modified = false;

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto& expr = *expr_it;
        const auto& convert = expr->get_node();
        if (!ov::is_type<ov::op::v0::Convert>(convert))
            continue;

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
