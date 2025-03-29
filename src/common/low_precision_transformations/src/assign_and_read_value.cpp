// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/assign_and_read_value.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "low_precision/fake_quantize.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

AssignAndReadValueTransformation::AssignAndReadValueTransformation(const std::shared_ptr<ov::Model> model, const Params& params) :
    LayerTransformation(params), model(model) {
    MATCHER_SCOPE(AssignAndReadValueTransformation);
    auto assign_m = pattern::wrap_type<opset3::Assign, opset6::Assign>({ pattern::wrap_type<ov::opset1::Multiply>() });

    ov::graph_rewrite_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto assign = m.get_match_root();
        // check that we have ReadValue as the first dependency
        if (assign->get_control_dependencies().empty()) {
            return false;
        }

        if (transformation_callback(assign)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(assign_m, matcher_name);
    this->register_matcher(m, callback);
}

bool AssignAndReadValueTransformation::transform(ov::pass::pattern::Matcher& m) {
    if (!canBeTransformed(m.get_match_root())) {
        return false;
    }

    const auto oldAssign = m.get_match_root();
    const auto readValue = oldAssign->get_control_dependencies()[0];
    oldAssign->remove_control_dependency(readValue);

    const auto assign = NetworkHelper::separateInStandaloneBranch(oldAssign, defaultPrecisions);
    const auto dequantization = NetworkHelper::getDequantization(assign, defaultPrecisions);

    auto oldVar = ov::as_type_ptr<op::util::ReadValueBase>(readValue)->get_variable();
    auto variableInfo = oldVar->get_info();
    // set new precision for oldVar to update precision in newReadValue
    oldVar->update({variableInfo.data_shape, dequantization.data.get_element_type(), variableInfo.variable_id});
    // transform ReadValue part
    const auto newConstant = foldConvert(readValue->get_input_node_shared_ptr(0), dequantization.data.get_element_type());
    const auto newReadValue = readValue->copy_with_new_inputs({newConstant});
    const auto newDequantization = dequantization.copyWithNewInput(newReadValue);
    replace_node(readValue, newDequantization);

    // transform Assign part

    const auto newAssign = assign->copy_with_new_inputs({dequantization.data});
    model->remove_sink(as_type_ptr<op::Sink>(oldAssign));
    model->add_sinks({as_type_ptr<op::Sink>(newAssign)});

    NetworkHelper::copyInfo(assign, newAssign);
    replace_node(assign, newAssign);
    newAssign->add_control_dependency(newReadValue);

    // fuse dequantization multiply with FQ after ReadValue if possible
    const auto nextLayers = newDequantization->get_output_target_inputs(0);
    if (nextLayers.size() > 1) {
        return true;
    }
    const auto fakeQuantize = as_type_ptr<ov::opset1::FakeQuantize>(nextLayers.begin()->get_node()->shared_from_this());

    if (fakeQuantize == nullptr) {
        return true;
    }
    auto fakeQuantizeInputs = fakeQuantize->input_values();

    const auto inputLow = as_type_ptr<ov::opset1::Constant>(fakeQuantizeInputs[1].get_node_shared_ptr());
    const auto inputHigh = as_type_ptr<ov::opset1::Constant>(fakeQuantizeInputs[2].get_node_shared_ptr());

    if (inputLow == nullptr || inputHigh == nullptr) {
        return true;
    }

    FakeQuantizeTransformation::fuseElementwise(this, fakeQuantize, updatePrecisions);

    return true;
}

bool AssignAndReadValueTransformation::canBeTransformed(const std::shared_ptr<Node>& op) const {
    if (!LayerTransformation::canBeTransformed(op)) {
        return false;
    }

    const auto readValue = ov::as_type_ptr<op::util::ReadValueBase>(op->get_control_dependencies()[0]);
    if (!readValue) {
        return false;
    }

    // TODO: remove this limitation and change the transformation when this constant will be accepted to be non-zero
    if (!NetworkHelper::isZeroConst(readValue->get_input_node_shared_ptr(0))) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    return dequantization.subtract == nullptr && dequantization.multiply != nullptr;
}

bool AssignAndReadValueTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
