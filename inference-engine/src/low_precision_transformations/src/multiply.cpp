// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/multiply.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MultiplyTransformation, "MultiplyTransformation", 0);

MultiplyTransformation::MultiplyTransformation(const Params& params) : EltwiseBaseTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Multiply>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "MultiplyTransformation");
    this->register_matcher(m, callback);
}

bool MultiplyTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    auto multiply = m.get_match_root();
    if (!LayerTransformation::canBeTransformed(context, multiply)) {
        return false;
    }

    NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(multiply, 0));
    NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(multiply, 1));

    multiply = NetworkHelper::separateInStandaloneBranch(multiply);
    auto newMultiply = multiply;

    auto fold_fake_quantizes = [](std::shared_ptr<Node>& multiply, const size_t index) {
        auto fakeQuantizeOnWeights = as_type_ptr<opset1::FakeQuantize>(multiply->get_input_node_shared_ptr(index));
        if (fakeQuantizeOnWeights != nullptr) {
            auto result = NetworkHelper::fold_fake_quantize(fakeQuantizeOnWeights);
            if (is_type<opset1::Constant>(result)) {
                replace_node(fakeQuantizeOnWeights, result);
            }
        }
    };

    fold_fake_quantizes(multiply, 0ul);
    fold_fake_quantizes(multiply, 1ul);

    const int fullPathIndex = getNotEmpty(multiply);
    if (fullPathIndex == -1) {
        const auto multiplyBranch = getMultiplyConstBranch(multiply);
        if (multiplyBranch.first != -1) {
            NetworkHelper::foldDequantization(multiply, multiplyBranch.first == 0 ? 1 : 0);
        }

        if (multiplyBranch.first == -1 || multiplyBranch.second == -1) {
            // constant folding on dequantization ops (for example: Convert on Subtract)
            NetworkHelper::foldDequantization(multiply, 0);
            NetworkHelper::foldDequantization(multiply, 1);
            return false;
        }

        auto multiplyParent = multiply->get_input_source_output(multiplyBranch.first);
        auto constParent = multiply->get_input_source_output(multiplyBranch.first == 0 ? 1 : 0);
        auto multiplyParentParent = multiplyParent.get_node_shared_ptr()->get_input_source_output(multiplyBranch.second);
        auto multiplyParentConst = multiplyParent.get_node_shared_ptr()->get_input_source_output(multiplyBranch.second == 0 ? 1 : 0);

        newMultiply = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 },
            std::vector<ngraph::element::Type>{ multiply->get_output_element_type(0) },
            ngraph::op::TemporaryReplaceOutputType(multiplyParentParent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(
                fold<opset1::Multiply>(
                    foldConvert(multiplyParentConst, element::f32),
                    foldConvert(constParent, element::f32)),
                element::f32).get());

        NetworkHelper::copyInfo(multiplyParent.get_node_shared_ptr(), newMultiply);
        NetworkHelper::copyInfo(multiply, newMultiply);

        if (!FakeQuantizeDequantization::checkElementwise(newMultiply)) {
            NetworkHelper::cleanRunTimeInfo(newMultiply);
        }
    } else {
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

        FakeQuantizeDequantization dequantizationEmptyPath = NetworkHelper::getDequantization(multiply, emptyPathIndex);
        if ((updatePrecisions && !dequantizationEmptyPath.empty() && !dequantizationEmptyPath.isLowPrecision()) ||
            (dequantizationEmptyPath.multiply == nullptr && dequantizationEmptyPath.subtract == nullptr)) {
            return false;
        }

        FakeQuantizeDequantization dequantizationFullPath = NetworkHelper::getDequantization(multiply, fullPathIndex);
        if (updatePrecisions && !dequantizationFullPath.empty() && !dequantizationFullPath.isLowPrecision()) {
            return false;
        }

        dequantizationEmptyPath = NetworkHelper::foldDequantization(multiply, emptyPathIndex);
        std::shared_ptr<Node> subtractValuesEmptyPath;
        std::shared_ptr<Node> multiplyValuesEmptyPath;
        std::tie(subtractValuesEmptyPath, multiplyValuesEmptyPath) = NetworkHelper::createEmptyValues(dequantizationEmptyPath);

        // check if empty path shifts are not zero
        if (!NetworkHelper::isZeroConst(subtractValuesEmptyPath)) {
            return false;
        }

        dequantizationFullPath = NetworkHelper::foldDequantization(multiply, fullPathIndex);
        std::shared_ptr<Node> subtractValuesFullPath;
        std::shared_ptr<Node> multiplyValuesFullPath;
        std::tie(subtractValuesFullPath, multiplyValuesFullPath) = NetworkHelper::createEmptyValues(dequantizationFullPath);


        // before: Y = (SC1 * (X1 - SH1)) * (SC2 * X2)
        // after : Y = (SC1' * (X1 - SH1)) * (X2) , where :
        //         SC1' = SC1 * SC2
        std::shared_ptr<Node> newMultiplyValuesFullPath = fold<opset1::Multiply>(multiplyValuesEmptyPath, multiplyValuesFullPath);
        OutputVector inputs{ {}, {} };
        inputs[emptyPathIndex] = dequantizationEmptyPath.data;
        inputs[fullPathIndex] = std::make_shared<DequantizationMultiply>(
            dequantizationFullPath.subtract == nullptr ?
                (dequantizationFullPath.convert == nullptr ?
                    dequantizationFullPath.data : dequantizationFullPath.convert) :
                dequantizationFullPath.subtract,
            newMultiplyValuesFullPath);

        newMultiply = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{ multiply->get_output_element_type(0) },
                ngraph::op::TemporaryReplaceOutputType(inputs[0], element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(inputs[1], element::f32).get());
        NetworkHelper::copyInfo(multiply, newMultiply);
    }

    replace_node(multiply, newMultiply);
    updateOutput(context, newMultiply, multiply);

    if (fullPathIndex != -1) {
        NetworkHelper::foldDequantization(newMultiply, fullPathIndex);
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
