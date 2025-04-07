// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_convert.hpp"

#include <memory>
#include <vector>

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/disable_cleanup_attribute.hpp"

#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FuseConvertTransformation::FuseConvertTransformation(const Params& params) : CleanupTransformation(params) {
    MATCHER_SCOPE(FuseConvertTransformation);
    auto multiply = pattern::wrap_type<ov::opset1::Multiply>({ pattern::wrap_type<ov::opset1::Convert>(), pattern::wrap_type<ov::opset1::Constant>() });
    auto subtract = pattern::wrap_type<ov::opset1::Subtract>({ pattern::wrap_type<ov::opset1::Convert>(), pattern::wrap_type<ov::opset1::Constant>() });
    auto add = pattern::wrap_type<ov::opset1::Add>({ pattern::wrap_type<ov::opset1::Convert>(), pattern::wrap_type<ov::opset1::Constant>() });
    auto fakeQuantize = pattern::wrap_type<ov::opset1::FakeQuantize>({
        pattern::wrap_type<ov::opset1::Convert>({pattern::wrap_type<ov::opset1::Constant>()}),
        pattern::any_input(),
        pattern::any_input(),
        pattern::any_input(),
        pattern::any_input()});
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(
        std::make_shared<pass::pattern::op::Or>(OutputVector{ multiply, subtract, add, fakeQuantize }),
        matcher_name);

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    this->register_matcher(matcher, callback);
}

namespace {

std::shared_ptr<Node> removeConvertIfPossibleForSubtract(
    const std::shared_ptr<ov::opset1::Convert>& convert,
    const std::shared_ptr<ov::opset1::Subtract>& subtract) {
    std::shared_ptr<Node> newSubtract;

    const element::Type precisionBeforeConvert = convert->input(0).get_element_type();
    if (NetworkHelper::checkConstantValuePrecision(precisionBeforeConvert, subtract->get_input_node_shared_ptr(1))) {
        newSubtract = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
            std::vector<ov::element::Type>{ element::f32, element::f32 }, std::vector<ov::element::Type>{},
            ov::op::TemporaryReplaceOutputType(convert->input_value(0), element::f32).get(),
            ov::op::TemporaryReplaceOutputType(subtract->input_value(1), element::f32).get());
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(newSubtract, subtract->get_output_element_type(0));
        replace_node(subtract, newSubtract);
    }

    return newSubtract;
}

} // namespace

bool FuseConvertTransformation::transform(ov::pass::pattern::Matcher &m) {
    const auto op = m.get_match_root();
    if (!canBeTransformed(op)) {
        return false;
    }

    const auto convert = ov::as_type_ptr<ov::opset1::Convert>(op->get_input_node_shared_ptr(0));
    auto parent = convert->input_value(0);

    if (ov::is_type<ov::opset1::Constant>(parent.get_node_shared_ptr())) {
        auto convertedConstant = foldConvert(parent, convert->get_convert_element_type());
        NetworkHelper::copyInfo(parent.get_node_shared_ptr(), convertedConstant);
        replace_node(convert, convertedConstant);
    } else {
        std::shared_ptr<Node> newOp;
        if (ov::is_type<ov::opset1::Subtract>(op)) {
            auto subtract = ov::as_type_ptr<ov::opset1::Subtract>(op);
            newOp = removeConvertIfPossibleForSubtract(convert, subtract);
        } else if (ov::is_type<ov::opset1::Multiply>(op)) {
            newOp = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                    std::vector<ov::element::Type>{ element::f32, element::f32 }, std::vector<ov::element::Type>{},
                    ov::op::TemporaryReplaceOutputType(convert->input_value(0), element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(op->input_value(1), element::f32).get());
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(newOp, op->get_output_element_type(0));
            replace_node(op, newOp);
        } else if (ov::is_type<ov::opset1::Add>(op)) {
            newOp = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Add>>(
                    std::vector<ov::element::Type>{ element::f32, element::f32 }, std::vector<ov::element::Type>{},
                    ov::op::TemporaryReplaceOutputType(convert->input_value(0), element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(op->input_value(1), element::f32).get());
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(newOp, op->get_output_element_type(0));
            replace_node(op, newOp);
        }

        if (newOp == nullptr) {
            return false;
        }

        ov::copy_runtime_info({ convert, op }, newOp);
        newOp->set_friendly_name(op->get_friendly_name());
        register_new_node(newOp);
    }

    return true;
}

bool FuseConvertTransformation::canBeTransformed(const std::shared_ptr<Node>& op) const {
    if (!CleanupTransformation::canBeTransformed(op)) {
        return false;
    }

    const auto convert = ov::as_type_ptr<ov::opset1::Convert>(op->get_input_node_shared_ptr(0));
    // issue #40395
    if (convert == nullptr) {
        return false;
    }

    const auto destType = convert->get_destination_type();
    if ((destType != element::f16) && (destType != element::f32)) {
        return false;
    }

    return true;
}

bool FuseConvertTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
