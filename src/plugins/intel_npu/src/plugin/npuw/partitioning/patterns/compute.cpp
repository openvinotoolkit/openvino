// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compute.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace compute {

namespace opp = ov::pass::pattern;

// TODO: visualize
DQMatMulGQu4::DQMatMulGQu4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto qweight = opp::wrap_type<ov::op::v0::Constant>();
    auto qzerop = opp::wrap_type<ov::op::v0::Constant>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();

    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});
    auto qcvtz = opp::wrap_type<ov::op::v0::Convert>({qzerop});
    auto qcvts = opp::wrap_type<ov::op::v0::Convert>({qcoeff});

    auto qsubz = opp::wrap_type<ov::op::v1::Subtract>({qcvtw, qcvtz});
    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qsubz, qcvts});

    auto qreshp = opp::wrap_type<ov::op::v1::Reshape>({qmuls, opp::any_input()});
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), qreshp});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_qweight = node_to_output.at(qweight).get_node_shared_ptr();
        auto matched_node_qzerop = node_to_output.at(qzerop).get_node_shared_ptr();
        auto matched_node_qcoeff = node_to_output.at(qcoeff).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qweight));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qzerop));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qcoeff));

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qweight);
        auto matched_qzerop = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qzerop);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qcoeff);

        if (ov::element::u4 == matched_qweight->get_element_type() &&
            ov::element::u4 == matched_qzerop->get_element_type() &&
            ov::element::f16 == matched_qcoeff->get_element_type()) {
            // Partitioning ignores Const->Convert nodes, so qcvtw, qcvtz and qcvts are not used
            auto matched_qsubz = node_to_output.at(qsubz).get_node_shared_ptr();
            auto matched_qmuls = node_to_output.at(qmuls).get_node_shared_ptr();
            auto matched_qreshp = node_to_output.at(qreshp).get_node_shared_ptr();
            auto matched_qmm = node_to_output.at(qmm).get_node_shared_ptr();

            node_to_gptr->at(matched_qsubz)->isolate(isol_tag);
            node_to_gptr->at(matched_qmuls)->isolate(isol_tag);
            node_to_gptr->at(matched_qreshp)->isolate(isol_tag);
            node_to_gptr->at(matched_qmm)->isolate(isol_tag);
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "TagDQMatMulGQu4"), std::move(callback));
}

// TODO: visualize
DQMatMulCWu4::DQMatMulCWu4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto qweight = opp::wrap_type<ov::op::v0::Constant>();
    auto qzerop = opp::wrap_type<ov::op::v0::Constant>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();

    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});
    auto qcvtz = opp::wrap_type<ov::op::v0::Convert>({qzerop});

    auto qsubz = opp::wrap_type<ov::op::v1::Subtract>({qcvtw, qcvtz});
    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qsubz, qcoeff});

    auto qcvtm = opp::wrap_type<ov::op::v0::Convert>({qmuls});
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), qcvtm});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_qweight = node_to_output.at(qweight).get_node_shared_ptr();
        auto matched_node_qzerop = node_to_output.at(qzerop).get_node_shared_ptr();
        auto matched_node_qcoeff = node_to_output.at(qcoeff).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qweight));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qzerop));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qcoeff));

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qweight);
        auto matched_qzerop = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qzerop);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qcoeff);

        if ((ov::element::u4 == matched_qweight->get_element_type() ||
             ov::element::u8 == matched_qweight->get_element_type()) &&
            (ov::element::u4 == matched_qzerop->get_element_type() ||
             ov::element::u8 == matched_qzerop->get_element_type()) &&
            ov::element::f16 == matched_qcoeff->get_element_type()) {
            // Partitioning ignores Const->Convert nodes, so qcvtw and qcvtz are not used
            auto matched_qsubz = node_to_output.at(qsubz).get_node_shared_ptr();
            auto matched_qmuls = node_to_output.at(qmuls).get_node_shared_ptr();
            auto matched_qcvtm = node_to_output.at(qcvtm).get_node_shared_ptr();
            auto matched_qmm = node_to_output.at(qmm).get_node_shared_ptr();

            node_to_gptr->at(matched_qsubz)->isolate(isol_tag);
            node_to_gptr->at(matched_qmuls)->isolate(isol_tag);
            node_to_gptr->at(matched_qcvtm)->isolate(isol_tag);
            node_to_gptr->at(matched_qmm)->isolate(isol_tag);
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "TagDQMatMulCWu4"), std::move(callback));
}

// TODO: visualize
DQMatMulGQi4::DQMatMulGQi4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto qweight = opp::wrap_type<ov::op::v0::Constant>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();

    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});

    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qcvtw, qcoeff});
    auto qreshp = opp::wrap_type<ov::op::v1::Reshape>({qmuls, opp::any_input()});
    auto qcvtr = opp::optional<ov::op::v0::Convert>({qreshp->output(0)});
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), qcvtr});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_qweight = node_to_output.at(qweight).get_node_shared_ptr();
        auto matched_node_qcoeff = node_to_output.at(qcoeff).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qweight));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qcoeff));

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qweight);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qcoeff);

        if ((ov::element::i4 == matched_qweight->get_element_type() ||
             ov::element::i8 == matched_qweight->get_element_type()) &&
            (ov::element::f16 == matched_qcoeff->get_element_type() ||
             ov::element::f32 == matched_qcoeff->get_element_type())) {
            // Partitioning ignores Const->Convert nodes, so qcvtw is not used
            auto matched_qmuls = node_to_output.at(qmuls).get_node_shared_ptr();
            auto matched_qreshp = node_to_output.at(qreshp).get_node_shared_ptr();
            auto matched_qmm = node_to_output.at(qmm).get_node_shared_ptr();

            node_to_gptr->at(matched_qmuls)->isolate(isol_tag);
            node_to_gptr->at(matched_qreshp)->isolate(isol_tag);
            node_to_gptr->at(matched_qmm)->isolate(isol_tag);

            auto qcvtr_iter = node_to_output.find(qcvtr);
            if (qcvtr_iter != node_to_output.end()) {
                auto matched_qcvtr = qcvtr_iter->second.get_node_shared_ptr();
                node_to_gptr->at(matched_qcvtr)->isolate(isol_tag);
            }
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "TagDQMatMulGQi4"), std::move(callback));
}

// TODO: visualize
DQMatMulCWi4::DQMatMulCWi4(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto qweight = opp::wrap_type<ov::op::v0::Constant>();
    auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();

    auto qcvtw = opp::wrap_type<ov::op::v0::Convert>({qweight});

    auto qmuls = opp::wrap_type<ov::op::v1::Multiply>({qcvtw, qcoeff});

    auto qcvtm = opp::wrap_type<ov::op::v0::Convert>({qmuls});
    auto qmm = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), qcvtm});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_qweight = node_to_output.at(qweight).get_node_shared_ptr();
        auto matched_node_qcoeff = node_to_output.at(qcoeff).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qweight));
        NPUW_ASSERT(ov::op::util::is_constant(matched_node_qcoeff));

        auto matched_qweight = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qweight);
        auto matched_qcoeff = std::static_pointer_cast<ov::op::v0::Constant>(matched_node_qcoeff);

        if ((ov::element::i4 == matched_qweight->get_element_type() ||
             ov::element::i8 == matched_qweight->get_element_type()) &&
            ov::element::f16 == matched_qcoeff->get_element_type()) {
            // Partitioning ignores Const->Convert nodes, so qcvtw is not used
            auto matched_qmuls = node_to_output.at(qmuls).get_node_shared_ptr();
            auto matched_qcvtm = node_to_output.at(qcvtm).get_node_shared_ptr();
            auto matched_qmm = node_to_output.at(qmm).get_node_shared_ptr();

            node_to_gptr->at(matched_qmuls)->isolate(isol_tag);
            node_to_gptr->at(matched_qcvtm)->isolate(isol_tag);
            node_to_gptr->at(matched_qmm)->isolate(isol_tag);
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(qmm, "TagDQMatMulCWi4"), std::move(callback));
}

// This is a case for Raw (f16/f32) MatMul connected directly to the Result.
//
// The following combinations are covered:
//
// act(f32)    -> MatMul(f32) -> Result
// weight(f32) ->
//
// act(f16)    -> MatMul(f16) -> to_f32 -> Result
// weight(f16) ->
//
// act(f32)    -> to_f16 -> MatMul -> to_f32 -> Result
// weight(f16) ----------->
//
// act(f32)    -----------> MatMul -> Result
// weight(f16) -- to_f32-->

VocabMatMul::VocabMatMul(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto act_in = opp::any_input();
    auto weight = opp::wrap_type<ov::op::v0::Constant>();

    auto ocvta = opp::optional<ov::op::v0::Convert>({act_in->output(0)});
    auto ocvtw = opp::optional<ov::op::v0::Convert>({weight->output(0)});

    auto mm = opp::wrap_type<ov::op::v0::MatMul>({ocvta, ocvtw});
    auto ocvtm = opp::optional<ov::op::v0::Convert>({mm->output(0)});

    auto res = opp::wrap_type<ov::op::v0::Result>({ocvtm});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_out_a = node_to_output.at(act_in).get_node_shared_ptr();
        auto matched_out_w = node_to_output.at(weight).get_node_shared_ptr();

        auto a_type = matched_out_a->get_element_type();
        auto w_type = matched_out_w->get_element_type();

        if ((a_type == ov::element::f16 || a_type == ov::element::f32) &&
            (w_type == ov::element::f16 || w_type == ov::element::f32)) {
            node_to_gptr->at(node_to_output.at(mm).get_node_shared_ptr())->isolate(isol_tag);

            auto isol_if = [=, &node_to_gptr, &node_to_output](std::shared_ptr<ov::Node> n) {
                auto iter = node_to_output.find(n);
                if (iter != node_to_output.end()) {
                    auto group_iter = node_to_gptr->find(iter->second.get_node_shared_ptr());
                    if (group_iter != node_to_gptr->end()) {
                        group_iter->second->isolate(isol_tag);
                    }
                }
            };
            isol_if(ocvta);
            isol_if(ocvtw);
            isol_if(ocvtm);
        }
        return false;
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "TagVocabMatMul"), std::move(callback));
}

// TODO: visualize
RMSNorm::RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto hadd = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), opp::any_input()});
    auto power = opp::wrap_type<ov::op::v1::Power>({hadd, opp::any_input()});
    auto reduce = opp::wrap_type<ov::op::v1::ReduceMean>({power, opp::any_input()});
    auto cadd = opp::wrap_type<ov::op::v1::Add>({reduce, opp::any_input()});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({cadd});
    auto div = opp::wrap_type<ov::op::v1::Divide>({opp::any_input(), sqrt});
    auto multiply1 = opp::wrap_type<ov::op::v1::Multiply>({hadd, div});
    auto multiply2 = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), multiply1});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_hadd = node_to_output.at(hadd).get_node_shared_ptr();
        auto matched_power = node_to_output.at(power).get_node_shared_ptr();
        auto matched_reduce = node_to_output.at(reduce).get_node_shared_ptr();
        auto matched_cadd = node_to_output.at(cadd).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();
        auto matched_div = node_to_output.at(div).get_node_shared_ptr();
        auto matched_multiply1 = node_to_output.at(multiply1).get_node_shared_ptr();
        auto matched_multiply2 = node_to_output.at(multiply2).get_node_shared_ptr();

        node_to_gptr->at(matched_hadd)->isolate(isol_tag);
        node_to_gptr->at(matched_power)->isolate(isol_tag);
        node_to_gptr->at(matched_reduce)->isolate(isol_tag);
        node_to_gptr->at(matched_cadd)->isolate(isol_tag);
        node_to_gptr->at(matched_sqrt)->isolate(isol_tag);
        node_to_gptr->at(matched_div)->isolate(isol_tag);
        node_to_gptr->at(matched_multiply1)->isolate(isol_tag);
        node_to_gptr->at(matched_multiply2)->isolate(isol_tag);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(multiply2, "TagRMSNorm"), std::move(callback));
}

}  // namespace compute
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
