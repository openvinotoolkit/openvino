// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_like.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace op {
namespace v1 {

ConvertLike::ConvertLike(const Output<Node>& data, const Output<Node>& like) : Op({data, like}) {
    constructor_validate_and_infer_types();
}

void ConvertLike::validate_and_infer_types() {
    OV_OP_SCOPE(v1_ConvertLike_validate_and_infer_types);
    set_output_type(0, get_input_element_type(1), get_input_partial_shape(0));
}

std::shared_ptr<Node> ConvertLike::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ConvertLike_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ConvertLike>(new_args.at(0), new_args.at(1));
}

bool ConvertLike::can_constant_fold(const OutputVector& input_values) const {
    return !is_const_fold_disabled();
}

bool ConvertLike::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v1_ConvertLike_constant_fold);
    if (!can_constant_fold(input_values)) {
        return false;
    }

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(input_values[0].get_node_shared_ptr())) {
        auto convert = std::make_shared<ov::op::v0::Convert>(input_values[0], input_values[1].get_element_type());
        return convert->constant_fold(output_values, OutputVector{data_const});
    }
    return false;
}

}  // namespace v1
}  // namespace op
}  // namespace ov
