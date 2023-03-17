// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "fuse_load_store_and_convert.hpp"
#include "snippets/snippets_isa.hpp"

#include "snippets_transformations/op/load_convert.hpp"
#include "snippets_transformations/op/store_convert.hpp"


bool ov::intel_cpu::pass::FuseLoadStoreConvert::fuse_load_convert(ngraph::snippets::LoweredExprIR& linear_ir,
                                                                  ngraph::snippets::LoweredExprIR::constExprIt& convert_it) {
    const auto& convert_expr = *convert_it;
    const auto& convert = ov::as_type_ptr<ov::op::v0::Convert>(convert_expr->get_node());
    const auto input_td = convert_expr->get_inputs().front();
    const auto output_td = convert_expr->get_outputs().front();
    if (convert->get_destination_type() != ov::element::f32 && convert->get_destination_type() != ov::element::i32)
        return false;

    const auto& load_output = linear_ir.get_expr_by_output(input_td);
    const auto& load_expr = load_output.expr;
    const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(load_expr->get_node());
    if (!load || load_expr->get_node()->get_type_info() != ngraph::snippets::op::Load::get_type_info_static())
        return false;

    const auto consumers = linear_ir.get_exprs_by_input(input_td);
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
        throw ov::Exception("Type of Convert op is undefined. Supports only fusing Load and ConvertTruncation or ConvertSaturation ops");
    }

    const auto in_td = std::vector<ngraph::snippets::TensorDescriptorPtr>{ load_expr->get_inputs().front() };
    const auto out_td = std::vector<ngraph::snippets::TensorDescriptorPtr>{ output_td };
    const auto mv_expr_it = convert_it;
    const auto& insertion_pos = std::next(convert_it);
    linear_ir.erase(std::find(linear_ir.cbegin(), mv_expr_it, load_expr));
    linear_ir.erase(mv_expr_it);
    convert_it = linear_ir.insert(insertion_pos, std::make_shared<ngraph::snippets::LoweredExpr>(load_convert, in_td, out_td));
    return true;
}

bool ov::intel_cpu::pass::FuseLoadStoreConvert::fuse_store_convert(ngraph::snippets::LoweredExprIR& linear_ir,
                                                                   ngraph::snippets::LoweredExprIR::constExprIt& convert_it) {
    const auto& convert_expr = *convert_it;
    const auto& convert = convert_expr->get_node();
    const auto input_td = convert_expr->get_inputs().front();
    const auto output_td = convert_expr->get_outputs().front();
    if (convert->get_input_element_type(0) != ov::element::f32 && convert->get_input_element_type(0) != ov::element::i32)
        return false;

    const auto consumers = linear_ir.get_exprs_by_input(output_td);
    if (consumers.size() != 1)
        return false;

    const auto store_input = *(consumers.begin());
    const auto store_expr = store_input.expr;
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
        throw ov::Exception("Type of Convert op is undefined. Supports only fusing Store and ConvertTruncation or ConvertSaturation ops");
    }

    const auto in_td = std::vector<ngraph::snippets::TensorDescriptorPtr>{ input_td };
    const auto out_td = std::vector<ngraph::snippets::TensorDescriptorPtr>{ store_expr->get_outputs().front() };
    const auto store_it = std::find(convert_it, linear_ir.cend(), store_expr);
    const auto& insertion_pos = std::next(store_it);
    linear_ir.erase(store_it);
    convert_it = linear_ir.erase(convert_it);
    linear_ir.insert(insertion_pos, std::make_shared<ngraph::snippets::LoweredExpr>(store_convert, in_td, out_td));
    return true;
}

bool ov::intel_cpu::pass::FuseLoadStoreConvert::run(ngraph::snippets::LoweredExprIR& linear_ir) {
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
