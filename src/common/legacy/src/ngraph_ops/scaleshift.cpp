// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/scaleshift.hpp"

#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

inline element::Type getMaxBitwidth(const std::vector<element::Type>& types) {
    if (types.empty()) {
        return element::undefined;
    }

    element::Type maxType = types[0];
    for (size_t i = 1; i < types.size(); ++i) {
        if (types[i].bitwidth() > maxType.bitwidth()) {
            maxType = types[i];
        }
    }
    return maxType;
}

BWDCMP_RTTI_DEFINITION(op::ScaleShiftIE);

op::ScaleShiftIE::ScaleShiftIE(const Output<Node>& data_batch, const Output<Node>& weights, const Output<Node>& bias, const element::Type output_type)
    : Op({data_batch, weights, bias}), output_type(output_type) {
    if (this->output_type == element::undefined) {
        this->output_type = getMaxBitwidth({ data_batch.get_element_type(), weights.get_element_type(), bias.get_element_type() });
    }
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::ScaleShiftIE::clone_with_new_inputs(const OutputVector& new_args) const {
    if (new_args.size() != 3) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<ScaleShiftIE>(new_args.at(0), new_args.at(1), new_args.at(2), output_type);
}

void op::ScaleShiftIE::validate_and_infer_types() {
    //  Check that weights and biases has the same type
    element::Type data_et = output_type == element::undefined ? get_input_element_type(0) : output_type;
    element::Type weights_et = get_input_element_type(1);
    element::Type biases_et = get_input_element_type(2);

    element::Type et_result;
    NODE_VALIDATION_CHECK(this, element::Type::merge(et_result, weights_et, biases_et),
                          "Element types for bias and weights do not match (biases element type: ", biases_et,
                          ", weights element type: ", weights_et, ").");

    set_output_type(0, data_et, get_input_partial_shape(0));
}

bool ngraph::op::ScaleShiftIE::visit_attributes(AttributeVisitor& visitor) {
    return true;
}
