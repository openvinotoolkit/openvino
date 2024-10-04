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

void Context::to_f16(PPtr orig_param) {
    closures_to_f16.insert(orig_param);
}

namespace opp = ov::pass::pattern;

// FROM:
//     ???(Act) ----------------------------------->
//     Param(W) -> to(f16) -> Multiply -> to(f32) -> MatMul
//     Param(S) ------------>
//
// TO:
//     ???(Act) -> to(f16) ->
//     Param(W) -> to(f16) -> MatMul -> Multiply -> to(f32)
//     Param(S) -> Reshape ----------->
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
//                 [32,1,128]              [1, 1 ,11K]        [32,1,11K]
//     ???(Act)  -> Reshape > Split(/32) ->[to(f16) ->       ]}
//     Param(W*) -----------> Split(/32) ->[to(f16) -> MatMul]} Concat v
//     Param(S)  ---------------------------------------------> Multiply
//                                                              Reshape(1,a,b,c)
//                                                              ReduceSum(1)
//                                                              Reshape(a,b,c)
//                                                              to(f32)
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
        auto out_shape = matched_node_matmul->output(0).get_shape();

        if (ov::element::i4 == matched_qweight->get_element_type() &&
            ov::element::f32 == matched_qcoeff->get_element_type() && qcoeff_shape.size() == 3 &&
            qweight_shape.size() == 3 && act_shape.size() == 3 && qcoeff_shape[0] == qweight_shape[0] &&
            qcoeff_shape[1] == 1 && qcoeff_shape[2] == qweight_shape[2] && !matched_matmul->get_transpose_a() &&
            !matched_matmul->get_transpose_b()) {
            // Mark W closure to transpose, and transpose the respective parameter
            ctx.get().permute(matched_qweight, {0, 2, 1});

            // Mark S closure to be lowered fo f16
            ctx.get().to_f16(matched_qcoeff);

            ov::Shape tw_shape = {qweight_shape[0], qweight_shape[2], qweight_shape[1]};
            matched_qweight->set_partial_shape(tw_shape);
            matched_qweight->validate_and_infer_types();

            matched_qcoeff->set_element_type(ov::element::f16);
            matched_qcoeff->validate_and_infer_types();

            // Reshape the Act to group format
            const auto NSPLIT = qweight_shape[0];
            std::vector<std::size_t> rshp_act_v = {NSPLIT, act_shape[1], act_shape[2] / NSPLIT};
            auto rshp_act_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, rshp_act_v);
            auto rshp_act = std::make_shared<ov::op::v1::Reshape>(matched_out_mmi, rshp_act_c, false);

            // Split Act and W, and S tensors by NSPLIT
            auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
            auto split_a = std::make_shared<ov::op::v1::Split>(rshp_act, split_axis, NSPLIT);
            auto split_w = std::make_shared<ov::op::v1::Split>(matched_qweight, split_axis, NSPLIT);

            // Do the CW MM for every split
            std::vector<std::shared_ptr<ov::Node>> to_concat;
            for (std::size_t i = 0; i < NSPLIT; i++) {
                auto a_f16 = std::make_shared<ov::op::v0::Convert>(split_a->output(i), ov::element::f16);
                auto w_f16 = std::make_shared<ov::op::v0::Convert>(split_w->output(i), ov::element::f16);
                auto m_f16 = std::make_shared<ov::op::v0::MatMul>(a_f16, w_f16, false, true);
                to_concat.push_back(m_f16);
            }

            // Now concat and scale the result
            auto concat = std::make_shared<ov::op::v0::Concat>(to_concat, 0);
            auto s_f16 = std::make_shared<ov::op::v1::Multiply>(concat, matched_qcoeff);

            // Now reshape to a better shape, ReduceSum, and reshape to the right size again
            std::vector<std::size_t> rshp_ccat_v = {1, NSPLIT, 1, qweight_shape[2]};
            auto rshp_ccat_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, rshp_ccat_v);
            auto rshp_ccat = std::make_shared<ov::op::v1::Reshape>(s_f16, rshp_ccat_c, false);

            auto reduce_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 1);
            auto reduce = std::make_shared<ov::op::v1::ReduceSum>(rshp_ccat, reduce_axis, true);

            auto rshp_out_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, out_shape);
            auto rshp_out = std::make_shared<ov::op::v1::Reshape>(reduce, rshp_out_c, false);

            // Convert the result to f32 to maintain the graph contracts. FIXME should be avoided
            auto out = std::make_shared<ov::op::v0::Convert>(rshp_out, ov::element::f32);

            // Now.. Reconnect the matmul readers to the new output (reducesum)
            for (auto&& r : matched_matmul->output(0).get_target_inputs()) {
                r.replace_source_output(out);
            }
            return true;  // root has changed
        }
        return false;  // did nothing here
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "OptDQMatMulGQi"), std::move(callback));
}

// FROM:
//     ???(Act) -------------------------------------------------------->
//     Param(W) -> Convert(f16) -> Multiply -> Reshape -> Convert(f32) -> MatMul
//     Param(S) ----------------->
//
// WHERE (example):
//     Act: [  1, 1,2048]
//     W:   [512,16, 128]
//     S:   [512,16,   1]
//                                         [1,  1,128]   x
// TO:                                     [1,512,128]T  =
//                 [16,1,128]              [1,  1,512]            [16,1,512]
//     ???(Act)  -> Reshape > Split(/16) ->[to(f16) ->          ]}
//     Param(W*) -----------> Split(/16) ->[to(f16) -> MatMul v ]} Concat
//     Param(S)  -----------> Split(/16) ->[---------> Multiply ]}   v
//                                                               Reshape(1,16,1,512)
//                                                               ReduceSum(1)
//                                                               Reshape(   1,1,512)
//                                                               to(f32)
// WHERE:
//     W* : [16,512,128]

DQMatMulGQ2i::DQMatMulGQ2i(Context::Ref ctx) {
    auto qweight = opp::wrap_type<ov::op::v0::Parameter>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Parameter>();
    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});
    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qcvtw, qcoeff});
    auto qreshp = opp::wrap_type<ov::op::v1::Reshape>({qmuls, opp::any_input()});
    auto qcvtr = opp::wrap_type<ov::op::v0::Convert>({qreshp});
    auto qmmi = opp::any_input();
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({qmmi, qcvtr});

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
        auto out_shape = matched_node_matmul->output(0).get_shape();

        if (ov::element::i4 == matched_qweight->get_element_type() && qweight_shape.size() == 3 &&
            ov::element::f16 == matched_qcoeff->get_element_type() && qcoeff_shape.size() == 3 &&
            act_shape.size() == 3 && qcoeff_shape[0] == qweight_shape[0] && qcoeff_shape[2] == 1 &&
            qcoeff_shape[1] == qweight_shape[1] && !matched_matmul->get_transpose_a() &&
            matched_matmul->get_transpose_b()) {
            // Mark W closure to transpose, and transpose the respective parameter
            ctx.get().permute(matched_qweight, {1, 0, 2});

            ov::Shape tw_shape = {qweight_shape[1], qweight_shape[0], qweight_shape[2]};
            matched_qweight->set_partial_shape(tw_shape);
            matched_qweight->validate_and_infer_types();

            // Reshape the Act to group format
            const auto NSPLIT = qweight_shape[1];
            std::vector<std::size_t> rshp_act_v = {NSPLIT, 1, act_shape[2] / NSPLIT};
            auto rshp_act_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, rshp_act_v);
            auto rshp_act = std::make_shared<ov::op::v1::Reshape>(matched_out_mmi, rshp_act_c, false);

            // Split Act and W, and S tensors by NSPLIT
            auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
            auto split_a = std::make_shared<ov::op::v1::Split>(rshp_act, split_axis, NSPLIT);
            auto split_w = std::make_shared<ov::op::v1::Split>(matched_qweight, split_axis, NSPLIT);

            auto split_axis_s = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 1);
            auto split_s = std::make_shared<ov::op::v1::Split>(matched_qcoeff, split_axis_s, NSPLIT);

            std::vector<std::size_t> rshp_scale_v = {1, 1, qcoeff_shape[0]};
            auto rshp_scale_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, rshp_scale_v);

            // Do the CW MM for every split
            std::vector<std::shared_ptr<ov::Node>> to_concat;
            for (std::size_t i = 0; i < NSPLIT; i++) {
                auto a_f16 = std::make_shared<ov::op::v0::Convert>(split_a->output(i), ov::element::f16);
                auto w_f16 = std::make_shared<ov::op::v0::Convert>(split_w->output(i), ov::element::f16);
                auto m_f16 = std::make_shared<ov::op::v0::MatMul>(a_f16, w_f16, false, true);

                auto r_f16 = std::make_shared<ov::op::v1::Reshape>(split_s->output(i), rshp_scale_c, false);
                auto s_f16 = std::make_shared<ov::op::v1::Multiply>(m_f16, r_f16);
                to_concat.push_back(s_f16);
            }

            // Now concat and scale the result
            auto concat = std::make_shared<ov::op::v0::Concat>(to_concat, 0);

            // Now reshape to a better shape, ReduceSum, and reshape to the right size again
            std::vector<std::size_t> rshp_ccat_v = {1, NSPLIT, 1, qweight_shape[0]};
            auto rshp_ccat_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, rshp_ccat_v);
            auto rshp_ccat = std::make_shared<ov::op::v1::Reshape>(concat, rshp_ccat_c, false);

            auto reduce_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 1);
            auto reduce = std::make_shared<ov::op::v1::ReduceSum>(rshp_ccat, reduce_axis, true);

            auto rshp_out_c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, out_shape);
            auto rshp_out = std::make_shared<ov::op::v1::Reshape>(reduce, rshp_out_c, false);

            // Convert the result to f32 to maintain the graph contracts. FIXME should be avoided
            auto out = std::make_shared<ov::op::v0::Convert>(rshp_out, ov::element::f32);

            // Now.. Reconnect the matmul readers to the new output (reducesum)
            for (auto&& r : matched_matmul->output(0).get_target_inputs()) {
                r.replace_source_output(out);
            }
            return true;  // root has changed
        }
        return false;  // did nothing here
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "OptDQMatMulGQ2i"), std::move(callback));
}

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
