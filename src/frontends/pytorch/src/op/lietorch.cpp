// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

Output<Node> mul(const NodeContext& ctx, const Output<Node>& a, const Output<Node>& b) {
    return ctx.mark_node(std::make_shared<v1::Multiply>(a, b));
}

Output<Node> sub(const NodeContext& ctx, const Output<Node>& a, const Output<Node>& b) {
    return ctx.mark_node(std::make_shared<v1::Subtract>(a, b));
}

Output<Node> add(const NodeContext& ctx, const Output<Node>& a, const Output<Node>& b) {
    return ctx.mark_node(std::make_shared<v1::Add>(a, b));
}

Output<Node> div(const NodeContext& ctx, const Output<Node>& a, const Output<Node>& b) {
    return ctx.mark_node(std::make_shared<v1::Divide>(a, b));
}

Output<Node> concat_last(const NodeContext& ctx, const OutputVector& items) {
    return ctx.mark_node(std::make_shared<v0::Concat>(items, -1));
}

Output<Node> i64c(const NodeContext& ctx, const Shape& shape, const std::vector<int64_t>& vals) {
    return ctx.mark_node(v0::Constant::create(element::i64, shape, vals));
}

// float constant matching the data tensor element type
Output<Node> fconst(const NodeContext& ctx, float value, const Output<Node>& like) {
    auto c = ctx.mark_node(v0::Constant::create(element::f32, Shape{}, {value}));
    return ctx.mark_node(std::make_shared<v1::ConvertLike>(c, like));
}

// gather one component along the last axis, shape (..., 1)
Output<Node> comp(const NodeContext& ctx, const Output<Node>& x, int64_t i) {
    return ctx.mark_node(std::make_shared<v8::Gather>(x, i64c(ctx, Shape{1}, {i}), i64c(ctx, Shape{}, {-1})));
}

// slice [lo:hi) along the last axis
Output<Node> range_last(const NodeContext& ctx, const Output<Node>& x, int64_t lo, int64_t hi) {
    auto start = i64c(ctx, Shape{1}, {lo});
    auto stop = i64c(ctx, Shape{1}, {hi});
    auto step = i64c(ctx, Shape{1}, {1});
    auto axes = i64c(ctx, Shape{1}, {-1});
    return ctx.mark_node(std::make_shared<v8::Slice>(x, start, stop, step, axes));
}

// cross product of two (..., 3) tensors along the last axis
Output<Node> cross_last(const NodeContext& ctx, const Output<Node>& a, const Output<Node>& b) {
    auto ax = comp(ctx, a, 0), ay = comp(ctx, a, 1), az = comp(ctx, a, 2);
    auto bx = comp(ctx, b, 0), by = comp(ctx, b, 1), bz = comp(ctx, b, 2);
    auto cx = sub(ctx, mul(ctx, ay, bz), mul(ctx, az, by));
    auto cy = sub(ctx, mul(ctx, az, bx), mul(ctx, ax, bz));
    auto cz = sub(ctx, mul(ctx, ax, by), mul(ctx, ay, bx));
    return concat_last(ctx, {cx, cy, cz});
}

// SE3 exp: tangent (..., 6) to SE3 data (..., 7)
// tau_phi = [tau(3) translation, phi(3) rotation], data layout [tx, ty, tz, qx, qy, qz, qw]
Output<Node> decompose_exp(const NodeContext& ctx, const Output<Node>& tau_phi) {
    auto tau = range_last(ctx, tau_phi, 0, 3);
    auto phi = range_last(ctx, tau_phi, 3, 6);

    // theta^2 as an explicit component sum, phi always has 3 elements
    auto px = comp(ctx, phi, 0), py = comp(ctx, phi, 1), pz = comp(ctx, phi, 2);
    auto theta2 = add(ctx, add(ctx, mul(ctx, px, px), mul(ctx, py, py)), mul(ctx, pz, pz));
    auto theta2_eps = add(ctx, theta2, fconst(ctx, 1e-12f, tau_phi));
    auto theta = ctx.mark_node(std::make_shared<v0::Sqrt>(theta2_eps));
    auto half_theta = mul(ctx, theta, fconst(ctx, 0.5f, tau_phi));

    // quaternion: q_xyz = sin(theta/2)/theta * phi, q_w = cos(theta/2)
    auto real = ctx.mark_node(std::make_shared<v0::Cos>(half_theta));
    auto imag = div(ctx, ctx.mark_node(std::make_shared<v0::Sin>(half_theta)), theta);
    auto q = concat_last(ctx, {mul(ctx, imag, phi), real});

    // translation: t = tau + c1*(phi x tau) + c2*(phi x (phi x tau))
    // c1 = (1 - cos theta) / theta^2, c2 = (theta - sin theta) / theta^3
    auto c1 =
        div(ctx, sub(ctx, fconst(ctx, 1.0f, tau_phi), ctx.mark_node(std::make_shared<v0::Cos>(theta))), theta2_eps);
    auto c2 = div(ctx, sub(ctx, theta, ctx.mark_node(std::make_shared<v0::Sin>(theta))), mul(ctx, theta2_eps, theta));
    auto phi_x_tau = cross_last(ctx, phi, tau);
    auto phi_x_phi_x_tau = cross_last(ctx, phi, phi_x_tau);
    auto t = add(ctx, tau, add(ctx, mul(ctx, c1, phi_x_tau), mul(ctx, c2, phi_x_phi_x_tau)));
    return concat_last(ctx, {t, q});
}

// SE3 act: SE3 data (..., 7) and points (..., 3) to (..., 3)
Output<Node> decompose_act(const NodeContext& ctx, const Output<Node>& data, const Output<Node>& p) {
    auto t = range_last(ctx, data, 0, 3);
    auto qv = range_last(ctx, data, 3, 6);
    auto qw = comp(ctx, data, 6);

    // rotate: u = 2*(qv x p); p_rot = p + qw*u + qv x u
    auto u = mul(ctx, fconst(ctx, 2.0f, data), cross_last(ctx, qv, p));
    auto p_rot = add(ctx, add(ctx, p, mul(ctx, qw, u)), cross_last(ctx, qv, u));
    return add(ctx, p_rot, t);
}

}  // namespace

// lietorch traces group ops as prim::PythonOp, leading inputs are tensor operands,
// any trailing inputs are closure captures and are ignored
OutputVector translate_lietorch_exp(const NodeContext& context) {
    num_inputs_check(context, 1, context.get_input_size());
    return {decompose_exp(context, context.get_input(0))};
};

OutputVector translate_lietorch_act3(const NodeContext& context) {
    num_inputs_check(context, 2, context.get_input_size());
    return {decompose_act(context, context.get_input(0), context.get_input(1))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
