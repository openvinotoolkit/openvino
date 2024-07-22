// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {

namespace opp = ov::pass::pattern;

//------------------------------------------------------------------------------
// Pattern: RMSNorm, from LLaMa-v2-7b model
//
//            Power     Const
//               :        :
//               V        V
//               ReduceMean
//                    :
//                    V
//                   Add
//                    :
//                    V
//                   Sqrt
//                    :
//                    V
//
RMSNormAvoid::RMSNormAvoid(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                           const std::string& avoid_device) {
    auto power = opp::wrap_type<ov::op::v1::Power>({opp::any_input(), opp::any_input()});
    auto reduce = opp::wrap_type<ov::op::v1::ReduceMean>({power, opp::wrap_type<ov::op::v0::Constant>()});
    auto add = opp::wrap_type<ov::op::v1::Add>({reduce, opp::any_input()});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({add});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_power = node_to_output.at(power).get_node_shared_ptr();
        auto matched_reduce = node_to_output.at(reduce).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();

        node_to_gptr->at(matched_power)->avoid(avoid_device);
        node_to_gptr->at(matched_reduce)->avoid(avoid_device);
        node_to_gptr->at(matched_add)->avoid(avoid_device);
        node_to_gptr->at(matched_sqrt)->avoid(avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sqrt, "TagRMSNormAvoid"), std::move(callback));
}

// TODO: visualize
DequantMatMulGQ::DequantMatMulGQ(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                 const std::string& isol_tag) {
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
    register_matcher(std::make_shared<opp::Matcher>(qmm, "TagDequantMatMulGQ"), std::move(callback));
}

// TODO: visualize
DequantMatMulCW::DequantMatMulCW(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                 const std::string& isol_tag) {
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

        if (ov::element::u4 == matched_qweight->get_element_type() &&
            ov::element::u4 == matched_qzerop->get_element_type() &&
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
    register_matcher(std::make_shared<opp::Matcher>(qmm, "TagDequantMatMulCW"), std::move(callback));
}

// TODO: visualize
SwishMultXMM::SwishMultXMM(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto swish = opp::wrap_type<ov::op::v4::Swish>({opp::any_input()});
    auto multiply = opp::wrap_type<ov::op::v1::Multiply>({swish, opp::any_input()});
    // FIXME: Match inputs/outputs against matmuls, but exclude those from the pattern

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_swish = node_to_output.at(swish).get_node_shared_ptr();
        auto matched_multiply = node_to_output.at(multiply).get_node_shared_ptr();

        node_to_gptr->at(matched_swish)->isolate(isol_tag);
        node_to_gptr->at(matched_multiply)->isolate(isol_tag);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(multiply, "TagSwishMultXMM"), std::move(callback));
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

// TODO: visualize
AdditionalCompute::AdditionalCompute(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                     const std::string& isol_tag) {
    auto shapeof = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shapeof, opp::any_input(), opp::any_input()});
    auto mod = opp::wrap_type<ov::op::v1::Mod>({gather, opp::any_input()});
    auto greater = opp::wrap_type<ov::op::v1::Greater>({mod, opp::any_input()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({greater});
    auto divide = opp::wrap_type<ov::op::v1::Divide>({gather, opp::any_input()});
    auto add = opp::wrap_type<ov::op::v1::Add>({divide, convert});
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({add, opp::any_input()});
    auto concat = opp::wrap_type<ov::op::v0::Concat>({broadcast, opp::any_input()});
    auto varsplit = opp::wrap_type<ov::op::v1::VariadicSplit>({opp::any_input(), opp::any_input(), concat});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_shapeof = node_to_output.at(shapeof).get_node_shared_ptr();
        auto matched_gather = node_to_output.at(gather).get_node_shared_ptr();
        auto matched_mod = node_to_output.at(mod).get_node_shared_ptr();
        auto matched_greater = node_to_output.at(greater).get_node_shared_ptr();
        auto matched_convert = node_to_output.at(convert).get_node_shared_ptr();
        auto matched_divide = node_to_output.at(divide).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
        auto matched_concat = node_to_output.at(concat).get_node_shared_ptr();
        auto matched_varsplit = node_to_output.at(varsplit).get_node_shared_ptr();

        node_to_gptr->at(matched_shapeof)->isolate(isol_tag);
        node_to_gptr->at(matched_gather)->isolate(isol_tag);
        node_to_gptr->at(matched_mod)->isolate(isol_tag);
        node_to_gptr->at(matched_greater)->isolate(isol_tag);
        node_to_gptr->at(matched_convert)->isolate(isol_tag);
        node_to_gptr->at(matched_divide)->isolate(isol_tag);
        node_to_gptr->at(matched_add)->isolate(isol_tag);
        node_to_gptr->at(matched_broadcast)->isolate(isol_tag);
        node_to_gptr->at(matched_concat)->isolate(isol_tag);
        node_to_gptr->at(matched_varsplit)->isolate(isol_tag);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(varsplit, "TagAdditionalCompute"), std::move(callback));
}

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
