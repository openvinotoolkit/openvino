// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/relu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;


OutputVector translate_prim_CallFunction(const NodeContext& context) {
    num_inputs_check(context, 2, context.get_input_size());

    auto function_input = context.get_input(0);

    // Get function arguments
    OutputVector args;
    for (size_t i = 1; i < context.get_input_size(); i++) {
        args.push_back(context.get_input(i));
    }

    Output<Node> result;

    if (auto const_op = std::dynamic_pointer_cast<v0::Constant>(function_input.get_node_shared_ptr())) {
        if (args.size() == 1) {
            auto arg_type = args[0].get_element_type();
            if (arg_type.is_signed()) {
                result = context.mark_node(std::make_shared<v0::Abs>(args[0]));
            } else {
                result = context.mark_node(std::make_shared<v0::Relu>(args[0]));
            }
        }

        else if (args.size() == 2) {
            result = args[0];
        }

        else {
            result = args[0];
        }
    } else {
        PYTORCH_OP_CONVERSION_CHECK(args.size() > 0, "prim::CallFunction: No arguments provided");
        result = args[0];
    }

    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        auto dtype = out_type.as<element::Type>();
        if (dtype.is_static() && dtype != result.get_element_type()) {
            result = context.mark_node(std::make_shared<v0::Convert>(result, dtype));
        }
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
