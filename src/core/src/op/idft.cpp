// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/idft.hpp"

#include <algorithm>
#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v7::IDFT);

op::v7::IDFT::IDFT(const Output<Node>& data, const Output<Node>& axes) : FFTBase(data, axes) {
    constructor_validate_and_infer_types();
}

op::v7::IDFT::IDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size)
    : FFTBase(data, axes, signal_size) {
    constructor_validate_and_infer_types();
}

bool op::v7::IDFT::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_IDFT_visit_attributes);
    return true;
}

std::shared_ptr<Node> op::v7::IDFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_IDFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 2) {
        return std::make_shared<op::v7::IDFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<op::v7::IDFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}
