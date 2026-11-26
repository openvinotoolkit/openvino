// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hyper_connection_post.hpp"

#include "itt.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::op::internal {

HyperConnectionPost::HyperConnectionPost(const Output<Node>& y,
                                         const Output<Node>& residual,
                                         const Output<Node>& post,
                                         const Output<Node>& comb)
    : Op({y, residual, post, comb}) {
    constructor_validate_and_infer_types();
}

void HyperConnectionPost::validate_and_infer_types() {
    OV_OP_SCOPE(internal_HyperConnectionPost_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "HyperConnectionPost expects 4 inputs");
    const auto et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this, et.is_dynamic() || et.is_real(), "residual must be floating point");
    const auto& rs = get_input_partial_shape(1);
    if (rs.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, rs.rank().get_length() >= 2, "residual must be [..., n, E]");
        const auto r = rs.rank().get_length();
        const auto& ys = get_input_partial_shape(0);
        if (ys.rank().is_static()) {
            NODE_VALIDATION_CHECK(this, ys.rank().get_length() == r - 1, "y must be [..., E]");
            NODE_VALIDATION_CHECK(this, ys[r - 2].compatible(rs[r - 1]), "y and residual widths differ");
        }
        const auto& cs = get_input_partial_shape(3);
        if (cs.rank().is_static()) {
            NODE_VALIDATION_CHECK(this, cs.rank().get_length() == r, "comb must be [..., n, n]");
            NODE_VALIDATION_CHECK(this,
                                  cs[r - 1].compatible(rs[r - 2]) && cs[r - 2].compatible(rs[r - 2]),
                                  "comb must be n x n");
        }
    }
    set_output_type(0, et, rs);
}

bool HyperConnectionPost::visit_attributes(AttributeVisitor&) {
    return true;
}

std::shared_ptr<ov::Node> HyperConnectionPost::clone_with_new_inputs(const ov::OutputVector& a) const {
    check_new_args_count(this, a);
    return std::make_shared<HyperConnectionPost>(a.at(0), a.at(1), a.at(2), a.at(3));
}

bool HyperConnectionPost::has_evaluate() const {
    const auto et = get_input_element_type(1);
    return et == ov::element::f32 || et == ov::element::f16;
}

bool HyperConnectionPost::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(internal_HyperConnectionPost_evaluate);
    const auto& rs = inputs[1].get_shape();
    const size_t E = rs.back(), n = rs[rs.size() - 2];
    size_t rows = 1;
    for (size_t i = 0; i + 2 < rs.size(); i++)
        rows *= rs[i];
    outputs[0].set_shape(rs);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* y = inputs[0].data<const T>();
        const T* res = inputs[1].data<const T>();
        const T* post = inputs[2].data<const T>();
        const T* comb = inputs[3].data<const T>();
        T* out = outputs[0].data<T>();
        for (size_t r = 0; r < rows; r++)
            for (size_t d = 0; d < n; d++)
                for (size_t e = 0; e < E; e++) {
                    float acc = static_cast<float>(y[r * E + e]) * static_cast<float>(post[r * n + d]);
                    for (size_t s = 0; s < n; s++)
                        acc += static_cast<float>(comb[(r * n + s) * n + d]) *
                               static_cast<float>(res[(r * n + s) * E + e]);
                    out[(r * n + d) * E + e] = static_cast<T>(acc);
                }
    };
    const auto et = inputs[1].get_element_type();
    if (et == ov::element::f32) {
        run(float{});
        return true;
    }
    if (et == ov::element::f16) {
        run(ov::float16{});
        return true;
    }
    return false;
}

}  // namespace ov::op::internal
