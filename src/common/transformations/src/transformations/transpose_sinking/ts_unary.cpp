// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_unary.hpp"

#include <utility>

#include "itt.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace v5 = ov::op::v5;
namespace v10 = ov::op::v10;
namespace op_util = ov::op::util;

namespace ov::pass {

namespace {

using NodePtr = std::shared_ptr<ov::Node>;

bool if_transpose_sinkable(const std::shared_ptr<v1::Transpose>& transpose,
                           const std::shared_ptr<v0::Constant>& transpose_order) {
    return static_cast<bool>(transpose);
}

}  // namespace

TSUnaryForward::TSUnaryForward() {
    MATCHER_SCOPE(TSUnaryForward);

    // We consider HardSigmoid, Swish, Selu, ConvertLike as unary ops
    // and handle only 0th input of these ops.
    create_pattern<op_util::UnaryElementwiseArithmetic,
                   v0::Clamp,
                   v0::Elu,
                   v4::SoftPlus,
                   v1::LogicalNot,
                   v0::Convert,
                   v10::IsInf,
                   v10::IsNaN,
                   v10::IsFinite,
                   v0::Selu,
                   v4::Swish,
                   v0::HardSigmoid,
                   v5::LogSoftmax,
                   v1::ConvertLike>({0}, if_transpose_sinkable);
    auto ts_unary_sinking_function = [this](const std::shared_ptr<Node>& main_node,
                                            const utils::TransposeInputsInfo& transpose_info) -> bool {
        bool res = utils::sink_forward::UpdateInputTransposes(main_node, transpose_info, {0});
        if (!res)
            return res;
        default_outputs_update(main_node, transpose_info);
        return true;
    };
    transpose_sinking(matcher_name, ts_unary_sinking_function);
}

TSUnaryBackward::TSUnaryBackward() {
    MATCHER_SCOPE(TSUnaryBackward);

    auto unary_restrictions = [](const Output<Node>& output) -> bool {
        return CheckTransposeConsumers(output);
    };

    auto unary_with_1_input_label = pattern::wrap_type<op_util::UnaryElementwiseArithmetic,
                                                       v0::Clamp,
                                                       v0::Elu,
                                                       v4::SoftPlus,
                                                       v1::LogicalNot,
                                                       v0::Convert,
                                                       v10::IsInf,
                                                       v10::IsNaN,
                                                       v10::IsFinite,
                                                       v5::LogSoftmax>({pattern::any_input()}, unary_restrictions);

    auto unary_with_2_inputs_label =
        pattern::wrap_type<v4::Swish, v1::ConvertLike>({pattern::any_input(), pattern::any_input()},
                                                       unary_restrictions);
    auto unary_with_3_inputs_label = pattern::wrap_type<v0::Selu, v0::HardSigmoid>(
        {pattern::any_input(), pattern::any_input(), pattern::any_input()},
        unary_restrictions);

    auto unary_label = std::make_shared<pattern::op::Or>(
        ov::OutputVector{unary_with_1_input_label, unary_with_2_inputs_label, unary_with_3_inputs_label});

    auto transpose_const_label = pattern::wrap_type<v0::Constant>();

    auto transpose_label = pattern::wrap_type<v1::Transpose>({unary_label, transpose_const_label});

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = transpose->get_input_node_shared_ptr(0);
        if (transformation_callback(unary)) {
            return false;
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(unary, transpose_const, {0})) {
            register_new_node(new_node);
        }
        unary->validate_and_infer_types();
        RemoveTransposeConsumers(unary);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, "ov::pass::TSUnaryBackward");
    register_matcher(m, matcher_pass_callback);
}

}  // namespace ov::pass
