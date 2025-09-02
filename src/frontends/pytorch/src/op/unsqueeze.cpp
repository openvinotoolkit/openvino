// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
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
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        const auto dim_const = ov::util::get_constant_from_source(dim);
        auto dim_data = dim_const->cast_vector<int64_t>();
        auto rank = std::get<1>(get_shape_rank(context, x, true));
        auto data = complex->get_input_source_output(0);

        bool all_negative = std::all_of(dim_data.begin(), dim_data.end(), [](int64_t v) {
            return v < 0;
        });
        dim = normalize_axis(context, dim, rank);

        if (all_negative) {
            auto one_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
            dim = context.mark_node(std::make_shared<v1::Add>(dim, one_1d));
        }
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