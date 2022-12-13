// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fold_fake_quantize.hpp"

#include <memory>
#include <string>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FoldFakeQuantizeTransformation::FoldFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FoldFakeQuantizeTransformation);
    auto matcher = std::make_shared<pattern::op::Or>(OutputVector {
            pattern::wrap_type<opset1::Multiply>(),
            pattern::wrap_type<opset1::FakeQuantize>()
        });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool FoldFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    const auto fakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
    if (fakeQuantize) {
        if (!canBeTransformed(context, fakeQuantize)) {
            return false;
        }

        const auto constantShape = fakeQuantize->input(1).get_partial_shape();
        if (constantShape.is_dynamic()) {
            return false;
        }

        std::shared_ptr<ngraph::Node> resultConstant = NetworkHelper::fold_fake_quantize(
            fakeQuantize,
            false,
            ((constantShape.rank().get_length() >= 2) && (constantShape[1] != 1ul)) ? 1ul : 0ul);
        if (ov::is_type<opset1::Constant>(resultConstant)) {
            replace_node(fakeQuantize, resultConstant);
            return true;
        }
    }

    //    y = Multiply(x, [0,0,....,0])
    // => y = Broadcast(Constant(0, typeof(y))), ShapeOf(x))
    const auto dequantMultiply = ov::as_type_ptr<opset1::Multiply>(m.get_match_root());
    if (dequantMultiply) {
        auto multiplyInputs = dequantMultiply->input_values();
        auto constScales = as_type_ptr<opset1::Constant>(multiplyInputs[0].get_node_shared_ptr());
        auto dataNode = multiplyInputs[1].get_node_shared_ptr();
        if (!constScales) {
            dataNode = multiplyInputs[0].get_node_shared_ptr();
            constScales = as_type_ptr<opset1::Constant>(multiplyInputs[1].get_node_shared_ptr());
        }

        if (constScales) {
            const auto scales = constScales->cast_vector<float>();
            if (std::all_of(scales.begin(), scales.end(), [](const float& v) {
                    return v == 0.0f;
                })) {
                const auto data_shape_node = std::make_shared<opset1::ShapeOf>(dataNode);
                const auto zero_const = std::make_shared<opset1::Constant>(dequantMultiply->get_output_element_type(0),
                                                                           Shape{},
                                                                           std::vector<size_t>({0}));
                std::shared_ptr<Node> resultConstant = std::make_shared<opset1::Broadcast>(zero_const, data_shape_node);
                replace_node(dequantMultiply, resultConstant);
                return true;
            }
        }
    }
    return false;
}

bool FoldFakeQuantizeTransformation::isConstantOutput(std::shared_ptr<ngraph::Node> node) const {
    const auto fakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(node);
    if (!fakeQuantize) {
        return false;
    }

    auto fakeQuantizeInputs = fakeQuantize->input_values();

    const auto outputLow = as_type_ptr<opset1::Constant>(fakeQuantizeInputs[3].get_node_shared_ptr());
    const auto outputHigh = as_type_ptr<opset1::Constant>(fakeQuantizeInputs[4].get_node_shared_ptr());

    if (outputLow == nullptr || outputHigh == nullptr) {
        return false;
    }

    const auto vecLow = outputLow->cast_vector<float>();
    const auto vecHigh = outputHigh->cast_vector<float>();

    return vecLow == vecHigh;
}

bool FoldFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    return NetworkHelper::isConstantPath(op) || isConstantOutput(op);
}

bool FoldFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
