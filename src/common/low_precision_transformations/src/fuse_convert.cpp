// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_convert.hpp"

#include <memory>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"
#include "low_precision/rt_info/skip_cleanup_attribute.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FuseConvertTransformation::FuseConvertTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FuseConvertTransformation);
    auto multiply = pattern::wrap_type<opset1::Multiply>({ pattern::wrap_type<opset1::Convert>(), pattern::wrap_type<opset1::Constant>() });
    auto subtract = pattern::wrap_type<opset1::Subtract>({ pattern::wrap_type<opset1::Convert>(), pattern::wrap_type<opset1::Constant>() });
    auto add = pattern::wrap_type<opset1::Add>({ pattern::wrap_type<opset1::Convert>(), pattern::wrap_type<opset1::Constant>() });
    auto fakeQuantize = pattern::wrap_type<opset1::FakeQuantize>({
        pattern::wrap_type<opset1::Convert>({pattern::wrap_type<opset1::Constant>()}),
        pattern::any_input(),
        pattern::any_input(),
        pattern::any_input(),
        pattern::any_input()});
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(
        std::make_shared<pattern::op::Or>(OutputVector{ multiply, subtract, add, fakeQuantize }),
        matcher_name);

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    this->register_matcher(matcher, callback);
}

namespace {

std::shared_ptr<Node> removeConvertIfPossibleForSubtract(
    const std::shared_ptr<opset1::Convert>& convert,
    const std::shared_ptr<opset1::Subtract>& subtract) {
    std::shared_ptr<Node> newSubtract;

    const element::Type precisionBeforeConvert = convert->input(0).get_element_type();
    if (NetworkHelper::checkConstantValuePrecision(precisionBeforeConvert, subtract->get_input_node_shared_ptr(1))) {
        newSubtract = std::make_shared<ngraph::op::TypeRelaxed<opset1::Subtract>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(convert->input_value(0), element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(subtract->input_value(1), element::f32).get());
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(newSubtract, subtract->get_output_element_type(0));
        replace_node(subtract, newSubtract);
    }

    return newSubtract;
}

} // namespace

bool FuseConvertTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    const auto op = m.get_match_root();
    if (!canBeTransformed(context, op)) {
        return false;
    }

    const auto convert = ov::as_type_ptr<opset1::Convert>(op->get_input_node_shared_ptr(0));
    auto parent = convert->input_value(0);

    if (ov::is_type<opset1::Constant>(parent.get_node_shared_ptr())) {
        auto convertedConstant = foldConvert(parent, convert->get_convert_element_type());
        NetworkHelper::copyInfo(parent.get_node_shared_ptr(), convertedConstant);
        replace_node(convert, convertedConstant);
    } else {
        std::shared_ptr<Node> newOp;
        if (ov::is_type<opset1::Subtract>(op)) {
            auto subtract = ov::as_type_ptr<opset1::Subtract>(op);
            newOp = removeConvertIfPossibleForSubtract(convert, subtract);
        } else if (ov::is_type<opset1::Multiply>(op)) {
            newOp = std::make_shared<ngraph::op::TypeRelaxed<opset1::Multiply>>(
                    std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
                    ngraph::op::TemporaryReplaceOutputType(convert->input_value(0), element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(op->input_value(1), element::f32).get());
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(newOp, op->get_output_element_type(0));
            replace_node(op, newOp);
        } else if (ov::is_type<opset1::Add>(op)) {
            newOp = std::make_shared<ngraph::op::TypeRelaxed<opset1::Add>>(
                    std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
                    ngraph::op::TemporaryReplaceOutputType(convert->input_value(0), element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(op->input_value(1), element::f32).get());
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(newOp, op->get_output_element_type(0));
            replace_node(op, newOp);
        }

        if (newOp == nullptr) {
            return false;
        }

        ngraph::copy_runtime_info({ convert, op }, newOp);
        newOp->set_friendly_name(op->get_friendly_name());
        register_new_node(newOp);
    }

    return true;
}

bool FuseConvertTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!getAttribute<SkipCleanupAttribute>(op).empty()) {
        return false;
    }

    const auto convert = ov::as_type_ptr<opset1::Convert>(op->get_input_node_shared_ptr(0));
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
} // namespace ngraph
