// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/eliminate_fake_quantize.hpp"

#include <memory>


#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "itt.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

EliminateFakeQuantizeTransformation::EliminateFakeQuantizeTransformation(const Params& params) : CleanupTransformation(params) {
    MATCHER_SCOPE(FuseMultiplyToFakeQuantizeTransformation);
    const auto matcher = pattern::wrap_type<ov::opset1::FakeQuantize>({
            pattern::any_input(),
            pattern::wrap_type<ov::opset1::Constant>(),
            pattern::wrap_type<ov::opset1::Constant>(),
            pattern::wrap_type<ov::opset1::Constant>(),
            pattern::wrap_type<ov::opset1::Constant>()
        });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        const auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    const auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool EliminateFakeQuantizeTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    const auto root = m.get_match_root();
    if (!canBeTransformed(context, root)) {
        return false;
    }

    return replace_output_update_name(root->output(0), root->input_value(0));
}

namespace {
bool check_interval(const std::shared_ptr<ov::opset1::FakeQuantize>& fq,
                    const std::shared_ptr<ov::opset1::Constant>& constant,
                    const float value,
                    const float max_diff,
                    const bool exact_comparison) noexcept {
    bool need_to_check_intervals = false;
    const auto& constant_values = constant->cast_vector<float>();
    for (const auto constant_value : constant_values) {
        if (std::fabs(constant_value - value) > std::numeric_limits<float>::epsilon()) {
            const auto diff = std::fabs(constant_value - value);
            if ((exact_comparison && (std::fabs(constant_value - value) > std::numeric_limits<float>::epsilon())) ||
                (diff > max_diff)) {
                return false;
            }

            need_to_check_intervals = true;
        }
    }

    if (need_to_check_intervals) {
        auto tmp_fq = as_type_ptr<ov::opset1::FakeQuantize>(fq->clone_with_new_inputs({
            constant,
            fq->get_input_node_shared_ptr(1),
            fq->get_input_node_shared_ptr(2),
            fq->get_input_node_shared_ptr(3),
            fq->get_input_node_shared_ptr(4)}));
        auto result = NetworkHelper::fold_fake_quantize(tmp_fq, false);
        const auto result_constant = as_type_ptr<ov::opset1::Constant>(result);
        if (result_constant == nullptr) {
            return false;
        }

        const auto& result_values = result_constant->cast_vector<float>();
        for (const auto result_value : result_values) {
            if (std::fabs(result_value - value) > std::numeric_limits<float>::epsilon()) {
                return false;
            }
        }
    }

    return true;
}

bool check_intervals(const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize) {
    const auto& element_type = fakeQuantize->get_output_element_type(0);
    const auto levels = fakeQuantize->get_levels();
    if (levels == 0) {
        return false;
    }
    const auto min_value = DataPrecision::getMinValue(element_type, levels);
    const auto max_value = DataPrecision::getMaxValue(element_type, levels);
    // let's divide before to avoid overflow
    const auto max_diff = max_value / levels - min_value / levels;
    // input intervals can be not equal with type intervals for low precision only
    const auto exact_comparison = !element_type.is_integral();

    return
        check_interval(fakeQuantize, ov::as_type_ptr<ov::opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(1)),
                       min_value, max_diff, exact_comparison) &&
        check_interval(fakeQuantize, ov::as_type_ptr<ov::opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(2)),
                       max_value, max_diff, exact_comparison) &&
        check_interval(fakeQuantize, ov::as_type_ptr<ov::opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(3)),
                       min_value, max_diff, true) &&
        check_interval(fakeQuantize, ov::as_type_ptr<ov::opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(4)),
                       max_value, max_diff, true);
}
} // namespace

bool EliminateFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!CleanupTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    const auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(operation);
    OPENVINO_ASSERT(fakeQuantize != nullptr, "unexpected operation type");

    const auto& input_type = fakeQuantize->get_input_element_type(0);
    const auto& output_type = fakeQuantize->get_output_element_type(0);
    if ((input_type != output_type) || (!check_intervals(fakeQuantize))) {
        return false;
    }

    return true;
}

bool EliminateFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
