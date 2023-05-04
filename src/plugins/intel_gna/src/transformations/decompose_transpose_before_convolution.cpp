// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/decompose_transpose_before_convolution.hpp"

#include "openvino/cc/pass/itt.hpp"
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <algorithm>
#include <memory>
#include <vector>

using ngraph::Node;
using ngraph::opset9::Add;
using ngraph::opset9::Concat;
using ngraph::opset9::Constant;
using ngraph::opset9::Convolution;
using ngraph::opset9::FakeQuantize;
using ngraph::opset9::Reshape;
using ngraph::opset9::StridedSlice;
using ngraph::opset9::Transpose;

namespace {

std::shared_ptr<Constant> make_constant(const std::vector<size_t>& data) {
    return std::make_shared<Constant>(ngraph::element::Type_t::i64, ngraph::Shape{data.size()}, data);
}

std::shared_ptr<Reshape> append_reshape(std::shared_ptr<Node> node, const std::vector<size_t>& data) {
    auto const_shape = make_constant(data);
    auto r = std::make_shared<Reshape>(node, const_shape, false);
    return r;
}

std::shared_ptr<Node> append_simple_transpose(std::shared_ptr<Node> node) {
    auto order = make_constant(ov::Shape{1, 0});
    auto t = std::make_shared<Transpose>(node, order);
    return t;
}

std::shared_ptr<Node> append_transpose(std::shared_ptr<Node> node) {
    auto shape_in = node->get_output_shape(0);
    bool left = true;
    auto total = shape_in[1] * shape_in[0];
    if (::std::min(shape_in[0], shape_in[1]) <= 8) {
        return append_simple_transpose(node);
    }
    if (shape_in[1] < shape_in[0]) {
        left = false;
    }
    if (left) {
        auto r1 = append_reshape(node, {8, total / 8});
        auto t1 = append_simple_transpose(r1);
        auto r2 = append_reshape(t1, {shape_in[0] / 8, shape_in[1] * 8});
        auto t2 = append_simple_transpose(r2);
        auto r3 = append_reshape(t2, {shape_in[1], shape_in[0]});
        return r3;
    } else {
        auto r1 = append_reshape(node, {total / 8, 8});
        auto t1 = append_simple_transpose(r1);
        auto r2 = append_reshape(t1, {shape_in[0] * 8, shape_in[1] / 8});
        auto t2 = append_simple_transpose(r2);
        auto r3 = append_reshape(t2, {shape_in[1], shape_in[0]});
        return r3;
    }
}

}  // namespace

static bool ConvertTranspose(std::shared_ptr<Node> convolution) {
    auto concat = convolution->get_input_node_ptr(0)->get_input_node_shared_ptr(0);
    auto shape = concat->get_output_shape(0);

    const size_t kSplitsSupported = 3;
    if (shape[2] != kSplitsSupported) {
        // TODO other cases not supported for now
        return false;
    }
    auto reshape_to_physical = ov::Shape{kSplitsSupported, shape[3], shape[1]};

    auto reshape_const = make_constant(reshape_to_physical);
    auto r = std::make_shared <Reshape>(concat, reshape_const, false);


    auto ssb1 = make_constant(ov::Shape{0, 0, 0});
    auto sse1 = make_constant(ov::Shape{1, shape[3], shape[1]});
    auto ssb2 = make_constant(ov::Shape{1, 0, 0});
    auto sse2 = make_constant(ov::Shape{2, shape[3], shape[1]});
    auto ssb3 = make_constant(ov::Shape{2, 0, 0});
    auto sse3 = make_constant(ov::Shape{3, shape[3], shape[1]});

    const std::vector<int64_t> mask{};

    auto ss1 = std::make_shared<StridedSlice>(r, ssb1, sse1, mask, mask);
    auto ss2 = std::make_shared<StridedSlice>(r, ssb2, sse2, mask, mask);
    auto ss3 = std::make_shared<StridedSlice>(r, ssb3, sse3, mask, mask);

    auto r1 = append_reshape(ss1, {shape[3], shape[1]});
    auto r2 = append_reshape(ss2, {shape[3], shape[1]});
    auto r3 = append_reshape(ss3, {shape[3], shape[1]});

    auto t1 = append_transpose(r1);
    auto t2 = append_transpose(r2);
    auto t3 = append_transpose(r3);

    auto r_2_1 = append_reshape(t1, {1, shape[1], shape[3]});
    auto r_2_2 = append_reshape(t2, {1, shape[1], shape[3]});
    auto r_2_3 = append_reshape(t3, {1, shape[1], shape[3]});
    ov::OutputVector c{r_2_1, r_2_2, r_2_3};
    auto new_concat = std::make_shared<Concat>(c, 0);

    auto final_reshape_node = append_reshape(new_concat, {1, shape[3], kSplitsSupported, shape[1]});

    convolution->input(0).replace_source_output(final_reshape_node);
    return true;
}

static bool ConvertTranspose2(std::shared_ptr<Node> reshape) {
    auto transpose = reshape->get_input_node_ptr(0);
    auto transpose_input = transpose->get_input_node_shared_ptr(0);
    const auto transpose_order_const = dynamic_cast<Constant*>(transpose->get_input_node_ptr(1));
    if (transpose_order_const == nullptr) {
        return false;
    }
    const auto transpose_order_value = transpose_order_const->cast_vector<int>();

    auto add_out_shape = transpose_input->get_output_shape(0);

    ov::Shape reshape_shape{add_out_shape[3], add_out_shape[1]};
    auto reshape_const = make_constant(reshape_shape);
    auto r = std::make_shared<Reshape>(transpose_input, reshape_const, false);
    auto t = append_transpose(r);
    ov::Shape reshape_shape2{1, add_out_shape[3], 1, add_out_shape[1]};
    auto reshape_const2 = make_constant(reshape_shape2);
    auto r2 = std::make_shared<Reshape>(t, reshape_const2, false);
    reshape->input(0).replace_source_output(r2);
    return true;
}

namespace ov {
namespace intel_gna {
namespace pass {
    using ngraph::pattern::any_input;
    using ngraph::pattern::wrap_type;
    using ngraph::pattern::Matcher;
DecomposeTranspose::DecomposeTranspose() {
    MATCHER_SCOPE(DecomposeTranspose);

    auto concat = wrap_type<Concat>();
    auto transpose = wrap_type<Transpose>({concat, any_input()});
    auto convolution = wrap_type<Convolution>({transpose, any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return ConvertTranspose(pattern_map.at(convolution).get_node_shared_ptr());
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

DecomposeTranspose2::DecomposeTranspose2() {
    MATCHER_SCOPE(DecomposeTranspose2);
    auto no_add = [](const Output<Node>& output) {
        return dynamic_cast<Add*>(output.get_node()) == nullptr;
    };

    auto constant_pattern = wrap_type<Constant>();
    auto add = wrap_type<Add>({any_input(no_add), any_input(no_add)});
    auto transpose = wrap_type<Transpose>({add, any_input()});
    auto reshape = wrap_type<Reshape>({transpose, constant_pattern});
    auto convolution = wrap_type<Convolution>({reshape, any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return ConvertTranspose2(pattern_map.at(reshape).get_node_shared_ptr());
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

DecomposeTranspose2FQ::DecomposeTranspose2FQ() {
    MATCHER_SCOPE(DecomposeTranspose2FQ);
    auto no_add = [](const Output<Node>& output) {
        return dynamic_cast<Add*>(output.get_node()) == nullptr;
    };

    auto constant_pattern = wrap_type<Constant>();
    auto fq1 = wrap_type<FakeQuantize>({any_input(no_add), any_input(), any_input(), any_input(), any_input()});
    auto fq2 = wrap_type<FakeQuantize>({any_input(no_add), any_input(), any_input(), any_input(), any_input()});
    auto add = wrap_type<Add>({fq1, fq2});
    auto fq = wrap_type<FakeQuantize>({add, any_input(), any_input(), any_input(), any_input()});
    auto transpose = wrap_type<Transpose>({fq, any_input()});
    auto reshape = wrap_type<Reshape>({transpose, constant_pattern});
    auto convolution = wrap_type<Convolution>({reshape, any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return ConvertTranspose2(pattern_map.at(reshape).get_node_shared_ptr());
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
