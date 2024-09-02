// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opt.hpp"

#include "../../logging.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
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

namespace opp = ov::pass::pattern;

// FROM:
//     Param(Act) ------------------------------------------->
//     Const(W) -> Convert(f16) -> Multiply -> Convert(f32) -> MatMul
//     Const(S) ----------------->
//
// TO:
//     Param(Act) -Convert(f16) ->
//     Const(W) -> Convert(f16) -> MatMul -> Multiply -> Convert(f32)
//     Const(S) -> Reshape ---------------->
//

DQMatMulCWi::DQMatMulCWi() {
    auto qweight = opp::wrap_type<ov::op::v0::Constant>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();
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

        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qweight));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qcoeff));

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qweight);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qcoeff);
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
            auto new_dims = std::vector<std::size_t>{ qcoeff_shape[1], qcoeff_shape[0] };
            auto new_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, new_dims);
            auto new_reshape = std::make_shared<ov::op::v1::Reshape>(matched_node_qcoeff, new_const, false);

            // Reconnect Multiply's both inputs. Drop all outputs
            matched_node_muls->input(0).replace_source_output(matched_matmul);
            matched_node_muls->input(1).replace_source_output(new_reshape);
            for (auto &&r : matched_node_muls->output(0).get_target_inputs()) {
                matched_node_muls->output(0).remove_target_input(r);
            }

            // Reconnect Convert(M) to convert the Multiply's result
            matched_node_cvtm->input(0).replace_source_output(matched_node_muls);

            // Reconnect MatMul's old readers to Convert(Multiply)
            for (auto &&r : mm_readers) {
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
// TO:
//                                <<<<<<<<<<<<<<< for each G >>>>>>>>>>>>>>
//     ???(Act) ----> Split(G) -->[to(f16) ->                             ]}
//     Param(W) ----> Split(G) -->[to(f16) -> MatMul -> to(f32) ->        ]} Concat -> ReduceSum
//     Param(S) ----> Split(G) -->[------------------------------>Multiply]}
//

DQMatMulGQi::DQMatMulGQi() {
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
        auto act_shape = matched_matmul->input(0).get_shape();

        if (ov::element::i4 == matched_qweight->get_element_type() &&
            qcoeff_shape.size() == 3 &&
            qweight_shape.size() == 3 &&
            act_shape.size() == 3 &&
            qcoeff_shape[0] == qweight_shape[0] &&
            qcoeff_shape[1] == 1 &&
            qcoeff_shape[2] == qweight_shape[2] &&
            !matched_matmul->get_transpose_a() && !matched_matmul->get_transpose_b()) {
            auto a_f16 = std::make_shared<ov::op::v0::Convert>(matched_out_mmi, ov::element::f16);
            auto w_f16 = std::make_shared<ov::op::v0::Convert>(matched_qweight, ov::element::f16);

            // Split Act, W, and S tensors by the group size
            const auto NSPLIT = qweight_shape[0];
            auto a_split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 2);
            auto w_split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
            auto s_split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
            auto split_a = std::make_shared<ov::op::v1::Split>(a_f16, a_split_axis, NSPLIT);
            auto split_w = std::make_shared<ov::op::v1::Split>(w_f16, w_split_axis, NSPLIT);
            auto split_s = std::make_shared<ov::op::v1::Split>(matched_node_qcoeff, s_split_axis, NSPLIT);

            // Do the CW MM for every group
            std::vector<std::shared_ptr<ov::Node> > to_concat;
            for (std::size_t i = 0; i < NSPLIT; i++) {
                auto m_f16 = std::make_shared<ov::op::v0::MatMul>(split_a->output(i),
                                                                  split_w->output(i));
                auto m_f32 = std::make_shared<ov::op::v0::Convert>(m_f16, ov::element::f32);
                auto s_f32 = std::make_shared<ov::op::v1::Multiply>(m_f32, split_s->output(i));
                to_concat.push_back(s_f32);
            }
#if 0
            // Concat the vector. Reduce didn't work so do adds
            auto ccat_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
            auto ccat = std::make_shared<ov::op::v0::Concat>(to_concat, 0);
            auto cred = std::make_shared<ov::op::v1::ReduceSum>(ccat, ccat_axis, true);
#endif
            // Reduce via ADD
            std::vector<ov::Output<ov::Node>> reduce_add;
            reduce_add.push_back(std::make_shared<ov::op::v1::Add>(to_concat[0], to_concat[1]));
            for (std::size_t i = 1; i < NSPLIT-1; i++) {
                reduce_add.push_back(std::make_shared<ov::op::v1::Add>(reduce_add[i-1], to_concat[i+1]));
            }
            auto &cred = reduce_add.back();

            // Now.. Reconnect the matmul readers to the new output (reducesum)
            for (auto &&r : matched_matmul->output(0).get_target_inputs()) {
                r.replace_source_output(cred);
            }
            return true;  // root has changed
        }
        return false; // did nothing here
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "OptDQMatMulGQi"), std::move(callback));
}

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
