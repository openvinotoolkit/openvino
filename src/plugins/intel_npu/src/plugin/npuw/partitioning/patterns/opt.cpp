// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opt.hpp"

#include "../../logging.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

// FIXME: It is probably much better to have these transformations done
// just once per function BODY
// On LLMs, it may reduce the # of such transformations by a factor of
// 20..30x.

namespace ov {
namespace npuw {
namespace patterns {
namespace opt {

void Context::permute(PPtr orig_param, const Context::Axes& order) {
    closures_to_permute[orig_param] = order;
}

namespace opp = ov::pass::pattern;

// FROM:
//     ???(Act) --------------------------------------------->
//     Param(W) -> Convert(f16) -> Multiply -> Convert(f32) -> MatMul
//     Param(S) ----------------->
//
// TO:
//     ???(Act) -> Convert(f16) ->
//     Param(W) -> Convert(f16) -> MatMul -> Multiply -> Convert(f32)
//     Param(S) -> Reshape ---------------->
//

DQMatMulCWi::DQMatMulCWi() {
    auto qweight = opp::wrap_type<ov::op::v0::Parameter>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Parameter>();
    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});
    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qcvtw, qcoeff});
    auto qcvtm = opp::wrap_type<ov::op::v0::Convert>({qmuls});
    auto qmmi = opp::any_input();
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({qmmi, qcvtm});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_qweight = node_to_output.at(qweight).get_node_shared_ptr();
        auto matched_node_qcoeff = node_to_output.at(qcoeff).get_node_shared_ptr();
        auto matched_node_matmul = node_to_output.at(qmm).get_node_shared_ptr();

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_qweight);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_qcoeff);
        auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);

        auto qcoeff_shape = matched_qcoeff->output(0).get_shape();

        if (ov::element::i4 == matched_qweight->get_element_type() && qcoeff_shape[1] == 1 &&
            !matched_matmul->get_transpose_a() && matched_matmul->get_transpose_b()) {
            auto matched_node_cvtw = node_to_output.at(qcvtw).get_node_shared_ptr();
            auto matched_node_cvtm = node_to_output.at(qcvtm).get_node_shared_ptr();
            auto matched_node_muls = node_to_output.at(qmuls).get_node_shared_ptr();
            auto matched_node_mmi = node_to_output.at(qmmi).get_node_shared_ptr();

            // Reconnect MatMul to read from Convert(W) directly.
            // Note: ACT is f32 so has to be converted too.
            auto new_cvt_act = std::make_shared<ov::op::v0::Convert>(matched_node_mmi, ov::element::f16);
            matched_matmul->input(0).replace_source_output(new_cvt_act);
            matched_matmul->input(1).replace_source_output(matched_node_cvtw);

            // Store MatMul's readers
            auto mm_readers = matched_matmul->output(0).get_target_inputs();

            // Introduce a Reshape to alter Scale factor's shape
            auto new_dims = std::vector<std::size_t>{qcoeff_shape[1], qcoeff_shape[0]};
            auto new_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, new_dims);
            auto new_reshape = std::make_shared<ov::op::v1::Reshape>(matched_node_qcoeff, new_const, false);

            // Reconnect Multiply's both inputs. Drop all outputs
            matched_node_muls->input(0).replace_source_output(matched_matmul);
            matched_node_muls->input(1).replace_source_output(new_reshape);
            for (auto&& r : matched_node_muls->output(0).get_target_inputs()) {
                matched_node_muls->output(0).remove_target_input(r);
            }

            // Reconnect Convert(M) to convert the Multiply's result
            matched_node_cvtm->input(0).replace_source_output(matched_node_muls);

            // Reconnect MatMul's old readers to Convert(Multiply)
            for (auto&& r : mm_readers) {
                r.replace_source_output(matched_node_cvtm);
            }
        }

        return true;  // root has changed
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "OptDQMatMulCWi"), std::move(callback));
}

// FROM:
//     ???(Act) -------------------------------------------->
//     Param(W) -> Convert(f16|f32) -> Multiply -> Reshape -> MatMul
//     Param(S) --------------------->
//
// WHERE (example):
//     Act: [ 1,  1, 4096]
//     W:   [32,128,11008]
//     S:   [32,  1,11008]
//                                         [1, 1 ,128]   x
// TO:                                     [1,11K,128]T  =
//                 [32,1,128]              [1, 1 ,11K]
//     ???(Act)  -> Reshape > Split(/32) ->[to(f16) ->                                 ]} x32
//     Param(W*) -----------> Split(/32) ->[to(f16) -> MatMul -> to(f32) ->            ]} Reduce(Add)
//     Param(S)  -------------Split(/32) ->[------------------------------> Multiply ->]}
//
// WHERE:
//     W* : [32,11008,128]

DQMatMulGQi::DQMatMulGQi(Context::Ref ctx) {
    auto qweight = opp::wrap_type<ov::op::v0::Parameter>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Parameter>();
    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});
    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qcvtw, qcoeff});
    auto qreshp = opp::wrap_type<ov::op::v1::Reshape>({qmuls, opp::any_input()});
    auto qmmi = opp::any_input();
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({qmmi, qreshp});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_qweight = node_to_output.at(qweight).get_node_shared_ptr();
        auto matched_node_qcoeff = node_to_output.at(qcoeff).get_node_shared_ptr();
        auto matched_node_matmul = node_to_output.at(qmm).get_node_shared_ptr();
        auto matched_out_mmi = node_to_output.at(qmmi);

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_qweight);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_qcoeff);
        auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);

        auto qweight_shape = matched_qweight->output(0).get_shape();
        auto qcoeff_shape = matched_qcoeff->output(0).get_shape();
        auto act_shape = matched_out_mmi.get_shape();

        if (ov::element::i4 == matched_qweight->get_element_type() && qcoeff_shape.size() == 3 &&
            qweight_shape.size() == 3 && act_shape.size() == 3 && qcoeff_shape[0] == qweight_shape[0] &&
            qcoeff_shape[1] == 1 && qcoeff_shape[2] == qweight_shape[2] && !matched_matmul->get_transpose_a() &&
            !matched_matmul->get_transpose_b()) {
            // Mark W closure to transpose, and transpose the respective parameter
            ctx.get().permute(matched_qweight, {0, 2, 1});

            ov::Shape tw_shape = {qweight_shape[0], qweight_shape[2], qweight_shape[1]};
            matched_qweight->set_partial_shape(tw_shape);
            matched_qweight->validate_and_infer_types();

            // Reshape the Act to group format
            const auto NSPLIT = qweight_shape[0];
            std::vector<std::size_t> rshp_act_v = {NSPLIT, act_shape[1], act_shape[2] / NSPLIT};
            auto rshp_act_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, rshp_act_v);
            auto rshp_act = std::make_shared<ov::op::v1::Reshape>(matched_out_mmi, rshp_act_c, false);

            // Split Act and W, and S tensors by NSPLIT
            auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
            auto split_a = std::make_shared<ov::op::v1::Split>(rshp_act, split_axis, NSPLIT);
            auto split_w = std::make_shared<ov::op::v1::Split>(matched_qweight, split_axis, NSPLIT);
            auto split_s = std::make_shared<ov::op::v1::Split>(matched_qcoeff, split_axis, NSPLIT);

            // Do the CW MM for every split
            std::vector<std::shared_ptr<ov::Node>> to_concat;
            for (std::size_t i = 0; i < NSPLIT; i++) {
                auto a_f16 = std::make_shared<ov::op::v0::Convert>(split_a->output(i), ov::element::f16);
                auto w_f16 = std::make_shared<ov::op::v0::Convert>(split_w->output(i), ov::element::f16);
                auto m_f16 = std::make_shared<ov::op::v0::MatMul>(a_f16, w_f16, false, true);
                auto m_f32 = std::make_shared<ov::op::v0::Convert>(m_f16, ov::element::f32);
                auto s_f32 = std::make_shared<ov::op::v1::Multiply>(m_f32, split_s->output(i));
                to_concat.push_back(s_f32);
            }

            std::vector<ov::Output<ov::Node>> reduce;
            reduce.push_back(std::make_shared<ov::op::v1::Add>(to_concat[0], to_concat[1]));
            for (std::size_t i = 1; i < NSPLIT - 1; i++) {
                reduce.push_back(std::make_shared<ov::op::v1::Add>(reduce[i - 1], to_concat[i + 1]));
            }
            auto sum = reduce.back();

            // Now.. Reconnect the matmul readers to the new output (reducesum)
            for (auto&& r : matched_matmul->output(0).get_target_inputs()) {
                r.replace_source_output(sum);
            }
            return true;  // root has changed
        }
        return false;  // did nothing here
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "OptDQMatMulGQi"), std::move(callback));
}

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
