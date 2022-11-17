// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/convert_like.hpp"

#include <memory>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::ConvertLike);

op::v1::ConvertLike::ConvertLike(const Output<Node>& data, const Output<Node>& like) : Op({data, like}) {
    constructor_validate_and_infer_types();
}

void op::v1::ConvertLike::validate_and_infer_types() {
    OV_OP_SCOPE(v1_ConvertLike_validate_and_infer_types);
    set_output_type(0, get_input_element_type(1), get_input_partial_shape(0));
}

bool op::v1::ConvertLike::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_ConvertLike_visit_attributes);
    return true;
}

shared_ptr<Node> op::v1::ConvertLike::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ConvertLike_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ConvertLike>(new_args.at(0), new_args.at(1));
}

bool op::v1::ConvertLike::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v1_ConvertLike_constant_fold);
    if (is_const_fold_disabled()) {
        return false;
    }

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(input_values[0].get_node_shared_ptr())) {
        auto convert = make_shared<ov::op::v0::Convert>(input_values[0], input_values[1].get_element_type());
        return convert->constant_fold(output_values, OutputVector{data_const});
    }
    return false;
}
