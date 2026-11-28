// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hyper_connection_pre.hpp"

#include "itt.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/hyper_connection_pre.hpp"

namespace ov::op::internal {

HyperConnectionPre::HyperConnectionPre(const Output<Node>& streams,
                                       const Output<Node>& fn,
                                       const Output<Node>& scale,
                                       const Output<Node>& base,
                                       int64_t n_streams,
                                       int64_t sinkhorn_iterations,
                                       float eps,
                                       float rms_eps)
    : Op({streams, fn, scale, base}),
      m_n_streams(n_streams),
      m_sinkhorn_iterations(sinkhorn_iterations),
      m_eps(eps),
      m_rms_eps(rms_eps) {
    constructor_validate_and_infer_types();
}

void HyperConnectionPre::validate_and_infer_types() {
    OV_OP_SCOPE(internal_HyperConnectionPre_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "HyperConnectionPre expects 4 inputs");
    NODE_VALIDATION_CHECK(this, m_n_streams > 0, "n_streams must be positive");
    NODE_VALIDATION_CHECK(this, m_sinkhorn_iterations >= 1, "sinkhorn_iterations must be at least 1");
    const auto et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this, et.is_dynamic() || et.is_real(), "streams must be a floating-point tensor");
    const auto fet = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this, fet.is_dynamic() || fet.is_real(), "fn must be a floating-point tensor");
    const int64_t n = m_n_streams, mix = (2 + n) * n;
    const auto& ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, ps.rank().is_dynamic() || ps.rank().get_length() >= 2, "streams must have rank >= 2");
    ov::Dimension e = ov::Dimension::dynamic();
    if (ps.rank().is_static()) {
        const auto r = ps.rank().get_length();
        NODE_VALIDATION_CHECK(this, ps[r - 2].compatible(n), "streams second-to-last dimension must be n = ", n);
        e = ps[r - 1];
    }
    const auto& fps = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, fps.rank().compatible(2), "fn must have rank 2");
    if (fps.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, fps[0].compatible(mix), "fn rows must be (2 + n) * n = ", mix);
        NODE_VALIDATION_CHECK(this, (e * n).compatible(fps[1]), "fn columns must be n * e");
    }
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(2).compatible(ov::PartialShape{3}), "scale must have shape [3]");
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(3).compatible(ov::PartialShape{mix}),
                          "base must have shape [(2 + n) * n]");
    ov::PartialShape y_shape = ps, vec_shape = ps, comb_shape = ps;
    if (ps.rank().is_static()) {
        const auto r = ps.rank().get_length();
        y_shape = ov::PartialShape(std::vector<ov::Dimension>(ps.begin(), ps.end() - 2));
        y_shape.push_back(e);
        vec_shape = ov::PartialShape(std::vector<ov::Dimension>(ps.begin(), ps.end() - 2));
        vec_shape.push_back(n);
        comb_shape = vec_shape;
        comb_shape.push_back(n);
        (void)r;
    }
    set_output_type(0, et, y_shape);
    set_output_type(1, et, vec_shape);
    set_output_type(2, et, comb_shape);
}

bool HyperConnectionPre::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_HyperConnectionPre_visit_attributes);
    visitor.on_attribute("n_streams", m_n_streams);
    visitor.on_attribute("sinkhorn_iterations", m_sinkhorn_iterations);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("rms_eps", m_rms_eps);
    return true;
}

std::shared_ptr<ov::Node> HyperConnectionPre::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_HyperConnectionPre_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<HyperConnectionPre>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                m_n_streams,
                                                m_sinkhorn_iterations,
                                                m_eps,
                                                m_rms_eps);
}

bool HyperConnectionPre::has_evaluate() const {
    const auto et = get_input_element_type(0), fet = get_input_element_type(1);
    const bool et_ok = et == ov::element::f32 || et == ov::element::f16;
    const bool fet_ok = fet == ov::element::f32 || fet == ov::element::f16 || fet == ov::element::bf16;
    return et_ok && fet_ok && get_input_element_type(2) == et && get_input_element_type(3) == et;
}

bool HyperConnectionPre::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(internal_HyperConnectionPre_evaluate);
    const auto& in_shape = inputs[0].get_shape();
    const size_t n = static_cast<size_t>(m_n_streams);
    const size_t r = in_shape.size(), e = in_shape[r - 1];
    size_t rows = 1;
    for (size_t i = 0; i + 2 < r; i++)
        rows *= in_shape[i];
    ov::Shape y_shape(in_shape.begin(), in_shape.end() - 2), vec_shape = y_shape, comb_shape;
    y_shape.push_back(e);
    vec_shape.push_back(n);
    comb_shape = vec_shape;
    comb_shape.push_back(n);
    outputs[0].set_shape(y_shape);
    outputs[1].set_shape(vec_shape);
    outputs[2].set_shape(comb_shape);
    const auto et = inputs[0].get_element_type(), fet = inputs[1].get_element_type();
    auto run = [&](auto t_tag, auto w_tag) {
        using T = decltype(t_tag);
        using W = decltype(w_tag);
        ov::reference::hyper_connection_pre<T, W>(inputs[0].data<const T>(),
                                                  inputs[1].data<const W>(),
                                                  inputs[2].data<const T>(),
                                                  inputs[3].data<const T>(),
                                                  outputs[0].data<T>(),
                                                  outputs[1].data<T>(),
                                                  outputs[2].data<T>(),
                                                  rows,
                                                  n,
                                                  e,
                                                  static_cast<size_t>(m_sinkhorn_iterations),
                                                  m_eps,
                                                  m_rms_eps);
    };
    auto by_weight = [&](auto t_tag) {
        if (fet == ov::element::f32)
            return run(t_tag, float{}), true;
        if (fet == ov::element::f16)
            return run(t_tag, ov::float16{}), true;
        if (fet == ov::element::bf16)
            return run(t_tag, ov::bfloat16{}), true;
        return false;
    };
    if (et == ov::element::f32)
        return by_weight(float{});
    if (et == ov::element::f16)
        return by_weight(ov::float16{});
    return false;
}

}  // namespace ov::op::internal
