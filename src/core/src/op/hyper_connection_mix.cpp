// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hyper_connection_mix.hpp"

#include "itt.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/hyper_connection_mix.hpp"

namespace ov::op::internal {

HyperConnectionMix::HyperConnectionMix(const Output<Node>& mixes,
                                       const Output<Node>& scale,
                                       const Output<Node>& base,
                                       int64_t n_streams,
                                       int64_t sinkhorn_iterations,
                                       float eps)
    : Op({mixes, scale, base}),
      m_n_streams(n_streams),
      m_sinkhorn_iterations(sinkhorn_iterations),
      m_eps(eps) {
    constructor_validate_and_infer_types();
}

void HyperConnectionMix::validate_and_infer_types() {
    OV_OP_SCOPE(internal_HyperConnectionMix_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 3, "HyperConnectionMix expects 3 inputs");
    NODE_VALIDATION_CHECK(this, m_n_streams > 0, "n_streams must be positive");
    NODE_VALIDATION_CHECK(this, m_sinkhorn_iterations >= 1, "sinkhorn_iterations must be at least 1");
    const auto et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this, et.is_dynamic() || et.is_real(), "mixes must be a floating-point tensor");
    const int64_t n = m_n_streams;
    const int64_t mix = (2 + n) * n;

    const auto& ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, ps.rank().is_dynamic() || ps.rank().get_length() >= 1, "mixes must have rank >= 1");
    if (ps.rank().is_static()) {
        const auto& last = ps[ps.rank().get_length() - 1];
        NODE_VALIDATION_CHECK(this,
                              last.compatible(mix),
                              "mixes last dimension must be (2 + n) * n = ",
                              mix,
                              ", got ",
                              last);
    }
    const auto& sps = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, sps.compatible(ov::PartialShape{3}), "scale must have shape [3]");
    const auto& bps = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this, bps.compatible(ov::PartialShape{mix}), "base must have shape [(2 + n) * n]");

    ov::PartialShape vec_shape = ps, comb_shape = ps;
    if (ps.rank().is_static()) {
        const auto r = ps.rank().get_length();
        vec_shape[r - 1] = n;
        comb_shape[r - 1] = n;
        comb_shape.push_back(n);
    }
    set_output_type(0, et, vec_shape);
    set_output_type(1, et, vec_shape);
    set_output_type(2, et, comb_shape);
}

bool HyperConnectionMix::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_HyperConnectionMix_visit_attributes);
    visitor.on_attribute("n_streams", m_n_streams);
    visitor.on_attribute("sinkhorn_iterations", m_sinkhorn_iterations);
    visitor.on_attribute("eps", m_eps);
    return true;
}

std::shared_ptr<ov::Node> HyperConnectionMix::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_HyperConnectionMix_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<HyperConnectionMix>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_n_streams,
                                                m_sinkhorn_iterations,
                                                m_eps);
}

bool HyperConnectionMix::has_evaluate() const {
    const auto et = get_input_element_type(0);
    return et == ov::element::f32 || et == ov::element::f16;
}

bool HyperConnectionMix::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(internal_HyperConnectionMix_evaluate);
    const auto& in_shape = inputs[0].get_shape();
    const size_t n = static_cast<size_t>(m_n_streams);
    size_t rows = 1;
    for (size_t i = 0; i + 1 < in_shape.size(); i++)
        rows *= in_shape[i];
    ov::Shape vec_shape = in_shape, comb_shape = in_shape;
    vec_shape.back() = n;
    comb_shape.back() = n;
    comb_shape.push_back(n);
    outputs[0].set_shape(vec_shape);
    outputs[1].set_shape(vec_shape);
    outputs[2].set_shape(comb_shape);
    const auto et = inputs[0].get_element_type();
    auto run = [&](auto tag) {
        using T = decltype(tag);
        ov::reference::hyper_connection_mix<T>(inputs[0].data<const T>(),
                                               inputs[1].data<const T>(),
                                               inputs[2].data<const T>(),
                                               outputs[0].data<T>(),
                                               outputs[1].data<T>(),
                                               outputs[2].data<T>(),
                                               rows,
                                               n,
                                               static_cast<size_t>(m_sinkhorn_iterations),
                                               m_eps);
    };
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
