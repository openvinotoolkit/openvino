// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lietorch_se3_decomposition.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass;

namespace {

// builds nodes and keeps them so the result can inherit the framework node rt info
struct Builder {
    NodeVector nodes;

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        nodes.push_back(node);
        return node;
    }

    // gather one component along the last axis, shape (..., 1)
    Output<Node> comp(const Output<Node>& x, int64_t i) {
        auto idx = make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{i});
        auto axis = make<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{-1});
        return make<v8::Gather>(x, idx, axis);
    }

    // slice [lo:hi) along the last axis
    Output<Node> range_last(const Output<Node>& x, int64_t lo, int64_t hi) {
        auto start = make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{lo});
        auto stop = make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{hi});
        auto step = make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
        auto axes = make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
        return make<v8::Slice>(x, start, stop, step, axes);
    }

    // cross product of two (..., 3) tensors along the last axis
    Output<Node> cross_last(const Output<Node>& a, const Output<Node>& b) {
        auto ax = comp(a, 0), ay = comp(a, 1), az = comp(a, 2);
        auto bx = comp(b, 0), by = comp(b, 1), bz = comp(b, 2);
        auto cx = make<v1::Subtract>(make<v1::Multiply>(ay, bz), make<v1::Multiply>(az, by));
        auto cy = make<v1::Subtract>(make<v1::Multiply>(az, bx), make<v1::Multiply>(ax, bz));
        auto cz = make<v1::Subtract>(make<v1::Multiply>(ax, by), make<v1::Multiply>(ay, bx));
        return make<v0::Concat>(OutputVector{cx, cy, cz}, -1);
    }

    // float constant matching the data tensor element type
    Output<Node> fconst(float value, const Output<Node>& like) {
        auto c = make<v0::Constant>(element::f32, Shape{}, std::vector<float>{value});
        return make<v1::ConvertLike>(c, like);
    }

    // SE3 exp: tangent (..., 6) to SE3 data (..., 7)
    // tau_phi = [tau(3) translation, phi(3) rotation]
    // SE3 data layout: [tx, ty, tz, qx, qy, qz, qw]
    Output<Node> decompose_exp(const Output<Node>& tau_phi) {
        auto tau = range_last(tau_phi, 0, 3);
        auto phi = range_last(tau_phi, 3, 6);

        auto axis_last = make<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{-1});
        auto theta2 = make<v1::ReduceSum>(make<v1::Multiply>(phi, phi), axis_last, true);
        auto theta2_eps = make<v1::Add>(theta2, fconst(1e-12f, tau_phi));
        auto theta = make<v0::Sqrt>(theta2_eps);
        auto half_theta = make<v1::Multiply>(theta, fconst(0.5f, tau_phi));

        // quaternion: q_xyz = sin(theta/2)/theta * phi, q_w = cos(theta/2)
        auto real = make<v0::Cos>(half_theta);
        auto imag = make<v1::Divide>(make<v0::Sin>(half_theta), theta);
        auto q_xyz = make<v1::Multiply>(imag, phi);
        auto q = make<v0::Concat>(OutputVector{q_xyz, real}, -1);

        // translation: t = J_l(phi) @ tau = tau + c1*(phi x tau) + c2*(phi x (phi x tau))
        // c1 = (1 - cos theta) / theta^2, c2 = (theta - sin theta) / theta^3
        auto c1 = make<v1::Divide>(make<v1::Subtract>(fconst(1.0f, tau_phi), make<v0::Cos>(theta)), theta2_eps);
        auto c2 =
            make<v1::Divide>(make<v1::Subtract>(theta, make<v0::Sin>(theta)), make<v1::Multiply>(theta2_eps, theta));
        auto phi_x_tau = cross_last(phi, tau);
        auto phi_x_phi_x_tau = cross_last(phi, phi_x_tau);
        auto t =
            make<v1::Add>(tau,
                          make<v1::Add>(make<v1::Multiply>(c1, phi_x_tau), make<v1::Multiply>(c2, phi_x_phi_x_tau)));
        return make<v0::Concat>(OutputVector{t, q}, -1);
    }

    // SE3 act: SE3 data (..., 7) and points (..., 3) to (..., 3)
    Output<Node> decompose_act(const Output<Node>& data, const Output<Node>& p) {
        auto t = range_last(data, 0, 3);
        auto qv = range_last(data, 3, 6);
        auto qw = comp(data, 6);  // (..., 1)

        // rotate: u = 2*(qv x p); p_rot = p + qw*u + qv x u
        auto u = make<v1::Multiply>(fconst(2.0f, data), cross_last(qv, p));
        auto p_rot = make<v1::Add>(make<v1::Add>(p, make<v1::Multiply>(qw, u)), cross_last(qv, u));
        return make<v1::Add>(p_rot, t);  // translate
    }
};

}  // namespace

LieTorchSE3Decomposition::LieTorchSE3Decomposition() {
    const auto fw_node = pattern::wrap_type<PtFrameworkNode>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto node = ov::as_type_ptr<PtFrameworkNode>(m.get_match_root());
        if (!node) {
            return false;
        }
        const auto op_type = node->get_op_type();
        const bool is_exp = (op_type == "lietorch::Exp");
        const bool is_act = (op_type == "lietorch::Act3");
        if (!is_exp && !is_act) {
            return false;
        }

        // lietorch group ops are traced as autograd prim::PythonOp nodes
        // the leading inputs are the tensor operands, any trailing inputs are
        // closure captures (the module self or unrelated params) and are ignored
        Builder b;
        Output<Node> result;
        if (is_exp) {
            // SE3 exp(tau_phi) to SE3 data
            if (node->get_input_size() < 1) {
                return false;
            }
            result = b.decompose_exp(node->input_value(0));
        } else {
            // SE3(data) act(points) to transformed points
            if (node->get_input_size() < 2) {
                return false;
            }
            result = b.decompose_act(node->input_value(0), node->input_value(1));
        }

        copy_runtime_info_and_name(node, b.nodes);
        result.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
        ov::replace_node(node, {result});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(fw_node, "ov::frontend::pytorch::pass::LieTorchSE3Decomposition");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
