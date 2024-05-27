// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/clamp.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

// Image format conversion based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/adjust_saturation_op.cc

// shared_ptr<Node> convert_rgb_to_hsv(shared_ptr<Node> images) {
shared_ptr<tuple<shared_ptr<Node>, shared_ptr<Node>, shared_ptr<Node>>> convert_rgb_to_hsv(shared_ptr<Node> images, element::Type type) {

    auto const_zero_f_ = make_shared<v0::Constant>(type, Shape{}, 0.0f);
    auto const_one_f_ = make_shared<v0::Constant>(type, Shape{}, 1.0f);
    auto const_six_f_ = make_shared<v0::Constant>(type, Shape{}, 6.0f);

    // Find max and min across channel axis. Max = Value (V)
    auto const_minus_one_i_1 = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    auto max_rgb = make_shared<v1::ReduceMax>(images, const_minus_one_i_1, true);
    auto min_rgb = make_shared<v1::ReduceMin>(images, const_minus_one_i_1, true);
    auto range = make_shared<v1::Subtract>(max_rgb, min_rgb);
    auto vv = max_rgb;

    // Compute Saturation (S)
    auto ss_ = make_shared<v1::Divide>(range, vv);
    auto ss = make_shared<v1::Select>(make_shared<v1::Greater>(vv, const_zero_f_), ss_, const_zero_f_);

    // Compute normalization factor (for Hue calculation)
    auto norm = make_shared<v1::Divide>(
        const_one_f_,
        make_shared<v1::Multiply>(const_six_f_, range)
    );

    // Split the image tensor into R, G, B channels
    auto const_minus_one_i = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto channels = make_shared<v1::Split>(images, const_minus_one_i, 3);
    auto r = channels->output(0);
    auto g = channels->output(1);
    auto b = channels->output(2);

    // Compute Hue (H)
    
    // Determine which component is the max (V) to compute Hue (H)
    auto r_eq_v = make_shared<v1::Equal>(r, vv);
    auto g_eq_v = make_shared<v1::Equal>(g, vv);
    // auto b_eq_v = make_shared<v1::Equal>(b, vv);

    // r == vv: hh = norm * (g - b)
    auto hue_case_r = make_shared<v1::Multiply>(norm, make_shared<v1::Subtract>(g, b));

    // g == vv: hh = norm * (b - r) + 2.0 / 6.0
    auto const_2_by_6 = make_shared<v0::Constant>(type, Shape{}, 2.0f / 6.0f);
    auto hue_case_g = make_shared<v1::Add>(
        make_shared<v1::Multiply>(norm, make_shared<v1::Subtract>(b, r)),
        const_2_by_6
    );

    // b == vv: hh = norm * (r - g) + 4.0 / 6.0
    auto const_4_by_6 = make_shared<v0::Constant>(type, Shape{}, 4.0f / 6.0f);
    auto hue_case_b = make_shared<v1::Add>(
        make_shared<v1::Multiply>(norm, make_shared<v1::Subtract>(r, g)),
        const_4_by_6
    );

    // Select hue based on the maximum component
    // Check if `r` is the max, otherwise check if `g` is the max, if not use `b`'s hue
    auto hh = make_shared<v1::Select>(
        r_eq_v,
        hue_case_r,  // Use hue_case_r if r is max
        make_shared<v1::Select>(
            g_eq_v,
            hue_case_g,  // Use hue_case_g if g is max
            hue_case_b   // Use hue_case_b otherwise (b is max)
        )
    );

    // range = 0.0: hh = 0
    auto hh_zero_range = make_shared<v1::Select>(make_shared<v1::Equal>(range, const_zero_f_), const_zero_f_, hh);

    // hh < 0.0: hh = hh + 1
    auto hh_final = make_shared<v1::Select>(make_shared<v1::Less>(hh, const_zero_f_), make_shared<v1::Add>(hh_zero_range, const_one_f_), hh_zero_range);


    return make_shared<tuple<shared_ptr<Node>, shared_ptr<Node>, shared_ptr<Node>>>(hh_final, ss, vv);
    // return make_shared<v0::Concat>(NodeVector{hh, vv, s}, -1); 
}

shared_ptr<Node> hsv_to_rgb(shared_ptr<Node> h, shared_ptr<Node> s, shared_ptr<Node> v, element::Type type) {
    auto const_six_f_ = make_shared<v0::Constant>(type, Shape{}, 6.0f);
    auto const_two_f_ = make_shared<v0::Constant>(type, Shape{}, 2.0f);
    auto const_one_f_ = make_shared<v0::Constant>(type, Shape{}, 1.0f);
    
    // c = s * v;
    auto c = make_shared<v1::Multiply>(s, v);
    // m = v - c;
    auto m = make_shared<v1::Subtract>(v, c);
    // dh = h * 6;
    auto dh = make_shared<v1::Multiply>(h, const_six_f_);

    // fmodu rounded to within [0, 2)
    auto fmodu = make_shared<v1::FloorMod>(dh, const_two_f_);

    //  x = c * (1 - std::abs(fmodu - 1));
    auto x = make_shared<v1::Multiply>(c, make_shared<v1::Subtract>(const_one_f_, make_shared<v0::Abs>(make_shared<v1::Subtract>(fmodu, const_one_f_))));

    
    // h_category = static_cast<int>(dh);
    auto h_category = make_shared<v0::Convert>(make_shared<v0::Floor>(dh), element::i64);

    auto zeros = make_shared<v0::Constant>(type, x->get_shape(), 0.0f);

    auto rr_options = NodeVector{c, x, zeros, zeros, x, c};
    auto gg_options = NodeVector{x, c, c, x, zeros, zeros};
    auto bb_options = NodeVector{zeros, zeros, x, c, c, x};

    auto rr_concat = make_shared<v0::Concat>(rr_options, -1); 
    auto gg_concat = make_shared<v0::Concat>(gg_options, -1);
    auto bb_concat = make_shared<v0::Concat>(bb_options, -1);
    
    // Use a gather operation to select the correct channel values based on h_category
    int batch_dim = s->get_shape().size() - 1;

    auto axis = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto rr = make_shared<v8::Gather>(rr_concat, h_category, axis, batch_dim);
    auto gg = make_shared<v8::Gather>(gg_concat, h_category, axis, batch_dim);
    auto bb = make_shared<v8::Gather>(bb_concat, h_category, axis, batch_dim);

    // Adding m to each component
    auto r = make_shared<v1::Add>(rr, m);
    auto g = make_shared<v1::Add>(gg, m);
    auto b = make_shared<v1::Add>(bb, m);

    // Return concatenated RGB 
    return make_shared<v0::Concat>(NodeVector{r, g, b}, -1); 
}

OutputVector translate_adjust_saturation_op(const NodeContext& node) {
    default_op_checks(node, 2, {"AdjustSaturation"});
    auto images = node.get_input(0);
    auto scale = node.get_input(1);
    auto node_name = node.get_name();

    auto type = images.get_element_type();

    auto hsv_components = convert_rgb_to_hsv(images.get_node_shared_ptr(), type);
    auto hh = get<0>(*hsv_components);
    auto ss = get<1>(*hsv_components);
    auto vv = get<2>(*hsv_components);

    scale = make_shared<v1::ConvertLike>(scale, images);

    auto ss_adjust = make_shared<v0::Clamp>(make_shared<v1::Multiply>(ss, scale), 0.0f, 1.0f);

    auto new_images = hsv_to_rgb(hh, ss_adjust, vv, type);

    auto adjust_saturation = new_images->output(0);

    set_node_name(node_name, adjust_saturation.get_node_shared_ptr());
    return {adjust_saturation};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

