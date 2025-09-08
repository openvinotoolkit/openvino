// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov;
using namespace ov::op;

OutputVector translate_bernoulli(const NodeContext& context) {
    // supported signatures:
    // 1. aten::bernoulli(input, *, Generator? generator=None) -> Tensor
    // 2. aten::bernoulli(input, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
    // 3. aten::bernoulli(input, float p, *, Generator? generator=None) -> Tensor
    num_inputs_check(context, 1, 3);
    uint64_t global_seed = 0;
    auto input = context.get_input(0);
    auto input_type = input.get_element_type().is_static() ? input.get_element_type() : element::f64;
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto probs_threshold = input;

    bool with_p = false;
    size_t gen_ind = 1;
    if (!context.input_is_none(1)) {
        auto p = context.get_input(1);
        with_p = p.get_element_type().is_real() ? true : false;
        if (with_p) {
            // need to override probs thresholds and samples type
            input_type = p.get_element_type().is_static() ? p.get_element_type() : element::f64;
            gen_ind = 2;
            probs_threshold = p;
        }
    }

    if (!context.input_is_none(gen_ind)) {
        // retrieve seed set to Generator
        auto gen_const = as_type_ptr<v0::Constant>(context.get_input(static_cast<int>(gen_ind)).get_node_shared_ptr());
        PYTORCH_OP_CONVERSION_CHECK(gen_const, "aten::bernoulli expects a constant representing a generator seed");
        auto seed = gen_const->cast_vector<uint64_t>();
        global_seed = seed.size() > 0 ? seed[0] : global_seed;
    }

    // generate tensor of input shape with elements sampled from u ~ RandomUniform(0, 1) distribution
    // I[u < input] will represent samples of bernoulli distribution with parameter `input`
    auto const_zero = context.mark_node(std::make_shared<v0::Constant>(input_type, Shape{}, 0.0f));
    auto const_one = context.mark_node(std::make_shared<v0::Constant>(input_type, Shape{}, 1.0f));
    auto ru_samples = context.mark_node(std::make_shared<v8::RandomUniform>(input_shape,
                                                                            const_zero,
                                                                            const_one,
                                                                            input_type,
                                                                            global_seed,
                                                                            0,
                                                                            PhiloxAlignment::PYTORCH));
    if (!input.get_element_type().is_static()) {
        ru_samples = context.mark_node(std::make_shared<v1::ConvertLike>(ru_samples, probs_threshold));
    }
    auto bernoulli_samples = context.mark_node(std::make_shared<v1::Less>(ru_samples, probs_threshold));

    if (!with_p && !context.input_is_none(2)) {
        auto out = context.get_input(2);
        bernoulli_samples = context.mark_node(std::make_shared<v1::ConvertLike>(bernoulli_samples, out));
        context.mutate_input(2, bernoulli_samples);
    } else {
        bernoulli_samples = context.mark_node(std::make_shared<v1::ConvertLike>(bernoulli_samples, input));
    }
    return {bernoulli_samples};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
