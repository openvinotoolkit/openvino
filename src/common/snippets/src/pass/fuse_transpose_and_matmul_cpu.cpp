// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/fuse_transpose_and_matmul_cpu.hpp"
#include "snippets/snippets_isa.hpp"

#include "snippets/utils.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
const std::set<std::vector<int>> FuseTransposeMatMulCPU::supported_cases = {{0, 2, 1, 3}};
FuseTransposeMatMulCPU::FuseTransposeMatMulCPU() {
    MATCHER_SCOPE(FuseTransposeMatMulCPU);
    auto transpose_is_supported = [](const Output<Node>& transpose_port) {
        const auto transpose_node = transpose_port.get_node_shared_ptr();
        // it's safe to do so because of the patterns we used. alternatively we can do it through pattern_values_map
        const auto& constant = as_type_ptr<ngraph::opset1::Constant>(transpose_node->get_input_node_shared_ptr(1));
        // if Transpose in and out layout is not empty => something was already fused on this port
        if (!utils::get_port_layout(transpose_port).empty() ||
            !utils::get_port_layout(transpose_port.get_node_shared_ptr()->input_value(0)).empty() ||
            constant->get_output_element_type(0) != ngraph::element::i32)
//            transpose_node->output(0).get_target_inputs().size() != 1)
            return false;
        const auto& transpose_order = constant->get_vector<int>();
        // todo: this limitation is due to the fact that offsets are calculated in Kernel, and the only way
        //  to calc them non-default way is to set Parameter rt_info field. This limitation can be removed if
        //  the rt_info is properly propagated to the corresponding parameter
        if (!is_type<ngraph::opset1::Parameter>(transpose_node->get_input_node_shared_ptr(0)) ||
            supported_cases.count(transpose_order) == 0)
            return false;
        return true;
    };
    auto constant = pattern::wrap_type<opset1::Constant>();
    auto transpose = pattern::wrap_type<opset1::Transpose>({pattern::any_input(), constant}, transpose_is_supported);
    auto transpose_matcher = std::make_shared<pattern::Matcher>(transpose);
    auto matmul_any = pattern::wrap_type<op::MatMulCPU>({pattern::any_input(), pattern::any_input()});

    auto matmul_in0 = pattern::wrap_type<op::MatMulCPU>({transpose, pattern::any_input()});
    auto matmul_in1 = pattern::wrap_type<op::MatMulCPU>({pattern::any_input(), transpose});
    auto matmul_out0 = pattern::wrap_type<opset1::Transpose>({matmul_any, constant});
    auto matmul_or_transpose = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{matmul_in0, matmul_in1, matmul_out0});

    auto callback = [](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseTransposeMatMulCPU")
        auto set_layout_from_order = [](const std::shared_ptr<opset1::Transpose>& node, const ov::Output<Node>& port) {
            const auto& const_order = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(1));
            const auto& transpose_order = const_order->get_vector<int>();
            std::vector<size_t> layout;
            std::copy(transpose_order.begin(), transpose_order.end(), std::back_inserter(layout));
            auto& rt_info = port.get_tensor_ptr()->get_rt_info();
            rt_info["Layout"] = layout;
        };
        auto matmul = as_type_ptr<op::MatMulCPU>(m.get_match_root());
        // Transpose on the MatMul's output
        if (!matmul) {
            matmul = as_type_ptr<op::MatMulCPU>(m.get_match_root()->get_input_node_shared_ptr(0));
            const auto& matmul_out = matmul->output(0);
            const auto& transpose_out = m.get_match_value();
            for (const auto& in : transpose_out.get_target_inputs())
                in.replace_source_output(matmul->output(0));
            set_layout_from_order(as_type_ptr<opset1::Transpose>(transpose_out.get_node_shared_ptr()), matmul_out);
        }
        for (int i = 0; i < matmul->get_input_size(); i++) {
            const auto& in_value = matmul->input_value(i);
            if (const auto& transpose = as_type_ptr<opset1::Transpose>(in_value.get_node_shared_ptr())) {
                set_layout_from_order(transpose, transpose->input_value(0));
                matmul->set_argument(i, transpose->input_value(0));
            }
        }
        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        matmul->validate_and_infer_types();
        return true;
    };
    register_matcher(std::make_shared<pattern::Matcher>(matmul_or_transpose, matcher_name), callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph