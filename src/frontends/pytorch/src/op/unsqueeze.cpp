// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_unsqueeze(const NodeContext& context) {
    num_inputs_check(context, 2, 2, true);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    
    auto complex = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    bool is_complex = complex != nullptr;
    
    if (is_complex) {
        auto data = complex->get_input_source_output(0);
        auto res = context.mark_node(std::make_shared<v0::Unsqueeze>(data, dim));
        res = context.mark_node(std::make_shared<ComplexTypeMark>(res));
        return {res};
    }
    
    auto res = context.mark_node(std::make_shared<v0::Unsqueeze>(x, dim));
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov