// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/avg_pool.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

AvgPoolTransformation::AvgPoolTransformation() : LayerTransformation(Params()) {
}

AvgPoolTransformation::AvgPoolTransformation(const Params& params) : LayerTransformation(params) {
    //auto matcher = make_op_pattern<opset1::AvgPool>({ make_op_label<opset1::Multiply>() });
    auto matcher = ngraph::pattern::wrap_type<opset1::AvgPool>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || m_transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "AvgPoolTransformation");
    this->register_matcher(m, callback);
}

void AvgPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::AvgPool>({ make_op_label<opset1::Multiply>() }));
}

bool getUpdatePrecision(const std::shared_ptr<Node>& node) {
    auto& rtInfo = node->get_rt_info();
    auto it = rtInfo.find(ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name);
    if (it == rtInfo.end()) {
        return false;
    }

    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionPreservedAttribute>>(it->second);
    return !attribute->get().sharedValue->value;
}

bool AvgPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> pooling = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    const bool updatePrecision = getUpdatePrecision(pooling);

    //const std::vector<std::shared_ptr<ngraph::Node>> children = getChildrenRecursivelyExceptPrecisionPreserved(pooling);

    //bool updatePrecision;

    //if ((children.size() == 1ul) && (!this->layerTransformationsManager->isQuantized(children[0]))) {
    //    updatePrecision = false;
    //} else {
    //    updatePrecision = false;
    //    // NOTE: This check was added for models that don't have FQ after AvgPool
    //    //       They will have transparent precision as it was in old LPT.
    //    for (const auto& child : children) {
    //        if (!is_type<opset1::FakeQuantize>(child)) {
    //            updatePrecision = true;
    //            break;
    //        }
    //    }
    //}

    moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling), updatePrecision);
    return true;
}

bool AvgPoolTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    auto dequantization = NetworkHelper::getDequantization(operation);

    return !!dequantization.multiply;
}

bool AvgPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    const std::vector<std::shared_ptr<ngraph::Node>> children = getChildrenRecursivelyExceptPrecisionPreserved(layer);
    // NOTE: This check was added for models that don't have FQ after AvgPool
    //       They will have transparent precision as it was in old LPT.
    for (const auto& child : children) {
        if (!is_type<opset1::FakeQuantize>(child)) {
            return true;
        }
    }
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
