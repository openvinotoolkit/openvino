// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/pass/align_element_type.hpp"
#include "snippets/utils.hpp"
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph/op/util/op_types.hpp"

#include <ngraph/rt_info.hpp>

namespace {

auto is_in_out_op(const std::shared_ptr<ov::Node>& n) -> bool {
    return ov::is_type<ov::op::v0::Parameter>(n)
        || ov::is_type<ov::op::v0::Constant>(n)
        || ov::is_type<ov::op::v0::Result>(n);
}

// At the moment Subgraph supports only Eltwise, Convert and FQ (which is decomposed into Eltwises and Convert)
// And only Eltwises supports execution only in "exec_type". So we can check op type from the opposite
auto op_supports_only_exec_type(const std::shared_ptr<ov::Node>& n) -> bool {
    return !ov::is_type<ov::op::v0::Convert>(n);
}

// Check if executable operation supports only execution element type f32
// NOTE: Executable op is node that isn't Parameter/Constant/Result
auto is_executable_op_only_on_exec_type(const std::shared_ptr<ov::Node>& n) -> bool {
    return op_supports_only_exec_type(n) && !is_in_out_op(n);
}

}  // namespace

ngraph::snippets::pass::AlignElementType::AlignElementType(const ov::element::Type exec_type) : exec_type(exec_type) { }

bool ngraph::snippets::pass::AlignElementType::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_FUNCTION_SCOPE(AlignElementType);

    auto insertConvert = [](const std::shared_ptr<ov::Node>& op, const size_t idx, const ov::element::Type& element_type) -> void {
        auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(op->input(idx).get_source_output(), element_type);
        ngraph::copy_runtime_info(op->get_input_node_shared_ptr(idx), convert);
        op->set_argument(idx, convert);
    };

    // NOTE: We don't call validate_and_infer_types() to avoid precision conflicts on inputs
    bool rewritten = false;
    auto ops = m->get_ordered_ops();
    for (auto& op : ops) {
        if (is_in_out_op(op) || ov::is_type<ov::op::v0::Convert>(op)) {
            continue;
        }

        if (op_supports_only_exec_type(op)) {
            for (auto i = 0; i < op->inputs().size(); i++) {
                auto shared_input = op->get_input_node_shared_ptr(i);
                auto existing_convert = ov::as_type_ptr<ov::op::v0::Convert>(shared_input);
                // We should insert Convert before Ops, which supports only exec element type, only when:
                //  - Input is Convert with unsupported destination type
                //  - Input is Op which support any element type
                // We couldn't unite these conditions and just check that element type isn't supported exec type
                // because we don't call validate_and_infer_types() so we don't know new precisions
                if ((existing_convert && existing_convert->get_destination_type() != exec_type) || (!is_executable_op_only_on_exec_type(shared_input))) {
                    insertConvert(op, i, exec_type);
                    rewritten |= true;
                }
            }
            if (auto tr_node = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(op)) {
                tr_node->set_overridden_output_type(exec_type, 0);
                rewritten |= true;
            }
        } else {  // branch for the Movement ops and MatMul ops in the future
            for (auto i = 0; i < op->inputs().size(); i++) {
                auto shared_input = op->get_input_node_shared_ptr(i);
                // it's original element type because we don't use validate_and_infer_type() anywhere
                const auto original_eltype = op->input(i).get_element_type();
                // If before op there is another op that doesn't support execution on original element type, we know that
                // before this op will be inserted reverse Convert to support execution on supported element type (first branch of condition).
                // So we should return original element type for operations that can support low precision
                if (is_executable_op_only_on_exec_type(shared_input) && original_eltype != exec_type) {
                    insertConvert(op, i, original_eltype);
                    rewritten |= true;
                }
            }
        }
    }

    return rewritten;
}

bool ngraph::snippets::pass::AlignElementType::opNeedsAlignElementType(const std::shared_ptr<ov::Node>& op, const ov::element::Type exec_type) {
    // At the moment Snippets support only Eltwise/Convert/FQ which one output so we can just call get_element_type()
    return is_executable_op_only_on_exec_type(op) && op->get_element_type() != exec_type;
}