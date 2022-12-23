// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_fake_quantize.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FuseFakeQuantizeTransformation::FuseFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FuseMultiplyToFakeQuantizeTransformation);
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

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

bool FuseFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    const auto root = m.get_match_root();
    if (!canBeTransformed(context, root)) {
        return false;
    }

    auto fakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(root);

    auto parent_output = fakeQuantize->get_input_source_output(0);
    parent_output.remove_target_input(fakeQuantize->input(0));

    for (auto& input : fakeQuantize->output(0).get_target_inputs()) {
        input.replace_source_output(parent_output);
    }

    return true;
}

namespace {
bool check_interval(const std::shared_ptr<opset1::Constant>& constant, const float value) noexcept {
    const auto& constant_values = constant->get_vector<float>();
    for (const auto constant_value : constant_values) {
        if (std::fabs(constant_value - value) > std::numeric_limits<float>::epsilon()) {
            return false;
        }
    }
    return true;
}

bool check_intervals(const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) {
    const auto element_type = fakeQuantize->output(0).get_element_type();
    const auto min_value = DataPrecision::getMinValue(element_type, fakeQuantize->get_levels());
    const auto max_value = DataPrecision::getMaxValue(element_type, fakeQuantize->get_levels());
    return
        check_interval(ov::as_type_ptr<opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(1)), min_value) &&
        check_interval(ov::as_type_ptr<opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(2)), max_value) &&
        check_interval(ov::as_type_ptr<opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(3)), min_value) &&
        check_interval(ov::as_type_ptr<opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(4)), max_value);
}
} // namespace

bool FuseFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    auto fakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(operation);
    if ((fakeQuantize == nullptr) ||
        (!ov::is_type<opset1::Constant>(fakeQuantize->get_input_node_ptr(1))) ||
        (!ov::is_type<opset1::Constant>(fakeQuantize->get_input_node_ptr(2))) ||
        (!ov::is_type<opset1::Constant>(fakeQuantize->get_input_node_ptr(3))) ||
        (!ov::is_type<opset1::Constant>(fakeQuantize->get_input_node_ptr(4)))) {
        return false;
    }

    const auto input_type = fakeQuantize->get_input_source_output(0).get_element_type();
    const auto output_type = fakeQuantize->output(0).get_element_type();
    if ((input_type != output_type) || (!check_intervals(fakeQuantize))) {
        return false;
    }

    return true;
}

bool FuseFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
