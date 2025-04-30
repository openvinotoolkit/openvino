// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avoid.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace avoid {

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
RMSNorm::RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
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

// From DeepSeek
SinCos::SinCos(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto param_62071 = opp::wrap_type<ov::op::v0::Parameter>();
    auto iids = opp::wrap_type<ov::op::v0::Parameter>();
    auto cvt_iids = opp::wrap_type<ov::op::v0::Convert>({iids});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({param_62071, cvt_iids});
    auto cvt_gather = opp::wrap_type<ov::op::v0::Convert>({gather});
    auto cvt_any = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto pow = opp::wrap_type<ov::op::v1::Power>({cvt_gather, cvt_any});
    auto reduce_mean = opp::wrap_type<ov::op::v1::ReduceMean>({pow});
    auto cvt_any_2 = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto add = opp::wrap_type<ov::op::v1::Add>({reduce_mean, cvt_any_2});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({add});
    auto cvt_any_3 = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto divide = opp::wrap_type<ov::op::v1::Divide>({cvt_any_3, sqrt});
    auto multiply = opp::wrap_type<ov::op::v1::Multiply>({cvt_gather, divide});
    auto param_61917 = opp::wrap_type<ov::op::v0::Parameter>();
    auto multiply_2 = opp::wrap_type<ov::op::v1::Multiply>({param_61917, multiply});
    auto shape_of = opp::wrap_type<ov::op::v0::ShapeOf>({multiply_2});

    auto param_61921 = opp::wrap_type<ov::op::v0::Parameter>();
    auto pids = opp::wrap_type<ov::op::v0::Parameter>();
    auto gathered_shapeof = opp::wrap_type<ov::op::v8::Gather>({shape_of});
    auto concatd_gather = opp::wrap_type<ov::op::v0::Concat>({gathered_shapeof});
    auto broadcast = opp::wrap_type<ov::op::v1::Broadcast>({param_61921, concatd_gather});
    auto unsqzed_pids = opp::wrap_type<ov::op::v0::Unsqueeze>({pids});
    auto cvt_pids = opp::wrap_type<ov::op::v0::Convert>({unsqzed_pids});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, cvt_pids});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul});
    auto concat = opp::wrap_type<ov::op::v0::Concat>({transpose});
    auto sin = opp::wrap_type<ov::op::v0::Sin>({concat});
    auto cos = opp::wrap_type<ov::op::v0::Cos>({concat});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();    
        auto matched_cvt_iids = node_to_output.at(cvt_iids).get_node_shared_ptr();
        auto matched_gather = node_to_output.at(gather).get_node_shared_ptr();
        auto matched_cvt_gather = node_to_output.at(cvt_gather).get_node_shared_ptr();
        auto matched_cvt_any = node_to_output.at(cvt_any).get_node_shared_ptr();
        auto matched_pow = node_to_output.at(pow).get_node_shared_ptr();
        auto matched_reduce_mean = node_to_output.at(reduce_mean).get_node_shared_ptr();
        auto matched_cvt_any_2 = node_to_output.at(cvt_any_2).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();
        auto matched_cvt_any_3 = node_to_output.at(cvt_any_3).get_node_shared_ptr();
        auto matched_divide = node_to_output.at(divide).get_node_shared_ptr();
        auto matched_multiply = node_to_output.at(multiply).get_node_shared_ptr();
        auto matched_multiply_2 = node_to_output.at(multiply_2).get_node_shared_ptr();
        auto matched_shape_of = node_to_output.at(shape_of).get_node_shared_ptr();

        auto matched_gathered_shapeof = node_to_output.at(gathered_shapeof).get_node_shared_ptr();
        auto matched_concatd_gather = node_to_output.at(concatd_gather).get_node_shared_ptr();
        auto matched_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
        auto matched_unsqzed_pids = node_to_output.at(unsqzed_pids).get_node_shared_ptr();
        auto matched_cvt_pids = node_to_output.at(cvt_pids).get_node_shared_ptr();
        auto matched_matmul = node_to_output.at(matmul).get_node_shared_ptr();
        auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
        auto matched_concat = node_to_output.at(concat).get_node_shared_ptr();
        auto matched_sin = node_to_output.at(sin).get_node_shared_ptr();
        auto matched_cos = node_to_output.at(cos).get_node_shared_ptr();


        node_to_gptr->at(matched_cvt_iids)->avoid(avoid_device);
        node_to_gptr->at(matched_gather)->avoid(avoid_device);
        node_to_gptr->at(matched_cvt_gather)->avoid(avoid_device);
        node_to_gptr->at(matched_cvt_any)->avoid(avoid_device);
        node_to_gptr->at(matched_pow)->avoid(avoid_device);
        node_to_gptr->at(matched_reduce_mean)->avoid(avoid_device);
        node_to_gptr->at(matched_cvt_any_2)->avoid(avoid_device);
        node_to_gptr->at(matched_add)->avoid(avoid_device);
        node_to_gptr->at(matched_sqrt)->avoid(avoid_device);
        node_to_gptr->at(matched_cvt_any_3)->avoid(avoid_device);
        node_to_gptr->at(matched_divide)->avoid(avoid_device);
        node_to_gptr->at(matched_multiply)->avoid(avoid_device);
        node_to_gptr->at(matched_multiply_2)->avoid(avoid_device);
        node_to_gptr->at(matched_shape_of)->avoid(avoid_device);

        node_to_gptr->at(matched_gathered_shapeof)->avoid(avoid_device);
        node_to_gptr->at(matched_concatd_gather)->avoid(avoid_device);
        node_to_gptr->at(matched_broadcast)->avoid(avoid_device);
        node_to_gptr->at(matched_unsqzed_pids)->avoid(avoid_device);
        node_to_gptr->at(matched_cvt_pids)->avoid(avoid_device);
        node_to_gptr->at(matched_matmul)->avoid(avoid_device);
        node_to_gptr->at(matched_transpose)->avoid(avoid_device);
        node_to_gptr->at(matched_concat)->avoid(avoid_device);
        node_to_gptr->at(matched_sin)->avoid(avoid_device);
        node_to_gptr->at(matched_cos)->avoid(avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(cos, "TagSinCos"), std::move(callback)); 
}
}  // namespace avoid
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
