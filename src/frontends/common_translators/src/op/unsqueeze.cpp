// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"
#include "common_translators.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace common_translators {

OutputVector translate_unsqueeze(const NodeContext& context) {
    num_inputs_check(context, 2, 2, true);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);

    auto complex = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    bool is_complex = complex != nullptr;

    if (is_complex) {
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }

        auto data = complex->get_input_source_output(0);

        auto one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));

        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));

        auto neg_dim = context.mark_node(std::make_shared<v1::Subtract>(dim, one));
        auto zero_cond = context.mark_node(std::make_shared<v1::Less>(dim, zero));
        auto new_dim = context.mark_node(std::make_shared<v1::Select>(zero_cond, neg_dim, dim));
        const auto dim_const = ov::util::get_constant_from_source(new_dim);
        auto res = context.mark_node(std::make_shared<v0::Unsqueeze>(data, dim_const));
        res = context.mark_node(std::make_shared<ComplexTypeMark>(res));
        return {res};
    }

    auto res = context.mark_node(std::make_shared<v0::Unsqueeze>(x, dim));
    return {res};
};

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov