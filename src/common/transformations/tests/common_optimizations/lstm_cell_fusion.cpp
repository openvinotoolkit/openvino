// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lstm_cell_fusion.hpp"

#include <random>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov;

using LSTMCellFusionParam = std::tuple<bool,  // true if second input to matmul is transposed
                                       int,   // rank of bias (B)
                                       int>;  // split axis

class LSTMCellFusionTestSuite : public testing::WithParamInterface<LSTMCellFusionParam>, public TransformationTestsF {};

TEST_P(LSTMCellFusionTestSuite, SubgraphFusedToLSTMCell) {
    const auto& param = GetParam();
    bool weights_transposed = std::get<0>(param);
    int B_rank = std::get<1>(param);
    int split_axis_value = std::get<2>(param);
    size_t input_size = 3;
    size_t hidden_size = 2;

    {
        auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, input_size});
        auto H = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto concat = std::make_shared<op::v0::Concat>(OutputVector{X, H}, 1);
        Shape WR_shape = weights_transposed ? Shape{4 * hidden_size, input_size + hidden_size}
                                            : Shape{input_size + hidden_size, 4 * hidden_size};
        std::vector<float> WR_values(shape_size(WR_shape));
        std::iota(WR_values.begin(), WR_values.end(), 0.0f);
        auto WR = op::v0::Constant::create(element::f32, WR_shape, WR_values);
        auto matmul = std::make_shared<op::v0::MatMul>(concat, WR, false, weights_transposed);
        Shape B_shape = B_rank == 2 ? Shape{1, 4 * hidden_size} : Shape{4 * hidden_size};
        std::vector<float> B_values(shape_size(B_shape));
        std::iota(B_values.begin(), B_values.end(), 0.0f);
        auto B = op::v0::Constant::create(element::f32, B_shape, B_values);
        auto biasadd = std::make_shared<op::v1::Add>(matmul, B);
        auto split_axis = op::v0::Constant::create(element::i32, Shape{}, {split_axis_value});
        auto split = std::make_shared<op::v1::Split>(biasadd, split_axis, 4 /* num splits */);
        auto it = std::make_shared<op::v0::Sigmoid>(split->output(0));
        auto ct = std::make_shared<op::v0::Tanh>(split->output(1));
        auto ft = std::make_shared<op::v0::Sigmoid>(
            std::make_shared<op::v1::Add>(split->output(2), op::v0::Constant::create(element::f32, Shape{1, 1}, {1})));
        auto ot = std::make_shared<op::v0::Sigmoid>(split->output(3));
        auto mul = std::make_shared<op::v1::Multiply>(it, ct);
        auto mul1 = std::make_shared<op::v1::Multiply>(ft, C);
        auto Ct = std::make_shared<op::v1::Add>(mul, mul1);
        auto Ht = std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::Tanh>(Ct), ot);
        auto C_abs = std::make_shared<op::v0::Abs>(Ct);
        auto H_abs = std::make_shared<op::v0::Abs>(Ht);
        model = std::make_shared<Model>(NodeVector{H_abs, C_abs}, ParameterVector{X, H, C});
        manager.register_pass<ov::pass::LSTMCellFusion>();
    }

    {
        auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, input_size});
        auto H = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto concat = std::make_shared<op::v0::Concat>(OutputVector{X, H}, 1);
        Shape W_shape{4 * hidden_size, input_size};
        Shape R_shape{4 * hidden_size, hidden_size};
        std::vector<float> W_values = weights_transposed
                                          ? std::vector<float>{20, 21, 22, 25, 26, 27, 0,  1,  2,  5,  6,  7,
                                                               10, 11, 12, 15, 16, 17, 30, 31, 32, 35, 36, 37}
                                          : std::vector<float>{4, 12, 20, 5, 13, 21, 0, 8,  16, 1, 9,  17,
                                                               2, 10, 18, 3, 11, 19, 6, 14, 22, 7, 15, 23};
        auto W = op::v0::Constant::create(element::f32, W_shape, W_values);
        std::vector<float> R_values =
            weights_transposed ? std::vector<float>{23, 24, 28, 29, 3, 4, 8, 9, 13, 14, 18, 19, 33, 34, 38, 39}
                               : std::vector<float>{28, 36, 29, 37, 24, 32, 25, 33, 26, 34, 27, 35, 30, 38, 31, 39};
        auto R = op::v0::Constant::create(element::f32, R_shape, R_values);
        Shape B_shape{4 * hidden_size};
        std::vector<float> B_values{5, 6, 0, 1, 2, 3, 6, 7};
        auto B = op::v0::Constant::create(element::f32, B_shape, B_values);
        auto lstm_cell = std::make_shared<op::v4::LSTMCell>(X,
                                                            H,
                                                            C,
                                                            W,
                                                            R,
                                                            B,
                                                            hidden_size,
                                                            std::vector<std::string>{"sigmoid", "tanh", "tanh"});
        auto C_abs = std::make_shared<op::v0::Abs>(lstm_cell->output(1));
        auto H_abs = std::make_shared<op::v0::Abs>(lstm_cell->output(0));
        model_ref = std::make_shared<Model>(NodeVector{H_abs, C_abs}, ParameterVector{X, H, C});
        manager.register_pass<ov::pass::LSTMCellFusion>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(LSTMCellFusion,
                         LSTMCellFusionTestSuite,
                         testing::Combine(testing::Values(false, true), testing::Values(1, 2), testing::Values(1, -1)));

using LSTMCellTFKerasFusionParam = std::tuple<std::string,  // f activation function
                                              std::string,  // g activation function
                                              std::string,  // h activation function
                                              size_t,       // input size
                                              size_t>;      // hidden size

class LSTMCellFusionWithSplitWeights : public testing::WithParamInterface<LSTMCellTFKerasFusionParam>,
                                       public TransformationTestsF {};

namespace {
void generate_weights_value(std::vector<float>& weights_value, const Shape& weights_shape) {
    weights_value.resize(shape_size(weights_shape));
    std::mt19937 rng(9812);
    std::uniform_real_distribution<float> distribution(-300, 300);
    for (size_t i = 0; i < weights_value.size(); ++i) {
        weights_value[i] = distribution(rng);
    }
}

void generate_weights(std::vector<float>& w,
                      std::vector<float>& r,
                      std::vector<float>& b,
                      size_t input_size,
                      size_t hidden_size) {
    Shape w_shape({hidden_size, input_size});
    Shape r_shape({hidden_size, hidden_size});
    Shape b_shape({hidden_size});

    generate_weights_value(w, w_shape);
    generate_weights_value(r, r_shape);
    generate_weights_value(b, b_shape);
}

ov::Output<ov::Node> get_activation_function(const std::string& activation_name, const ov::Output<ov::Node>& input) {
    if (activation_name == "relu") {
        return std::make_shared<op::v0::Relu>(input);
    } else if (activation_name == "tanh") {
        return std::make_shared<op::v0::Tanh>(input);
    } else if (activation_name == "sigmoid") {
        return std::make_shared<op::v0::Sigmoid>(input);
    }

    throw "unsupported activation function";
}

ov::Output<ov::Node> generate_gate_subgraph(const std::shared_ptr<ov::Node>& x,
                                            const std::shared_ptr<ov::Node>& h,
                                            const std::vector<float>& w,
                                            const std::vector<float>& r,
                                            const std::vector<float>& b,
                                            size_t input_size,
                                            size_t hidden_size,
                                            const std::string& f_activation) {
    // w must be of a shape [input_size, hidden_size]
    // r must be of a shape [hidden_size, hidden_size]
    // b must be of a shape [hidden_size]
    Shape w_shape({input_size, hidden_size});
    Shape r_shape({hidden_size, hidden_size});
    Shape b_shape({hidden_size});

    auto w_const = op::v0::Constant::create(element::f32, w_shape, w);
    auto r_const = op::v0::Constant::create(element::f32, r_shape, r);
    auto b_const = op::v0::Constant::create(element::f32, b_shape, b);

    auto x_by_wi = std::make_shared<op::v0::MatMul>(x, w_const);
    auto x_by_wi_biased = std::make_shared<op::v1::Add>(x_by_wi, b_const);
    auto h_by_ri = std::make_shared<op::v0::MatMul>(h, r_const);
    auto it = std::make_shared<op::v1::Add>(x_by_wi_biased, h_by_ri);
    return get_activation_function(f_activation, it);
}

ov::Output<ov::Node> prepare_weight_fico(const std::vector<float>& f_val,
                                         const std::vector<float>& i_val,
                                         const std::vector<float>& c_val,
                                         const std::vector<float>& o_val,
                                         Shape w_shape) {
    auto f = std::make_shared<ov::op::v0::Constant>(element::f32, w_shape, f_val);
    auto i = std::make_shared<ov::op::v0::Constant>(element::f32, w_shape, i_val);
    auto c = std::make_shared<ov::op::v0::Constant>(element::f32, w_shape, c_val);
    auto o = std::make_shared<ov::op::v0::Constant>(element::f32, w_shape, o_val);

    auto tr_order = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
    auto f_tr = std::make_shared<ov::op::v1::Transpose>(f, tr_order);
    auto i_tr = std::make_shared<ov::op::v1::Transpose>(i, tr_order);
    auto c_tr = std::make_shared<ov::op::v1::Transpose>(c, tr_order);
    auto o_tr = std::make_shared<ov::op::v1::Transpose>(o, tr_order);

    ov::Output<ov::Node> w = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{f_tr, i_tr, c_tr, o_tr}, 0);
    if (const auto& constant = ov::util::constantfold_subgraph(w)) {
        w = constant;
    }

    return w;
}
}  // namespace

TEST_P(LSTMCellFusionWithSplitWeights, SubgraphFusedToLSTMCell) {
    const auto& param = GetParam();
    const std::string& f_activation = std::get<0>(param);
    const std::string& g_activation = std::get<1>(param);
    const std::string& h_activation = std::get<2>(param);
    size_t input_size = std::get<3>(param);
    size_t hidden_size = std::get<4>(param);
    size_t batch_size = 2;

    // generate weights values
    std::vector<float> wi, ri, bi, wf, rf, bf, wo, ro, bo, wc, rc, bc;
    generate_weights(wi, ri, bi, input_size, hidden_size);
    generate_weights(wf, rf, bf, input_size, hidden_size);
    generate_weights(wo, ro, bo, input_size, hidden_size);
    generate_weights(wc, rc, bc, input_size, hidden_size);

    {
        auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
        auto h = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
        auto c = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});

        auto it = generate_gate_subgraph(x, h, wi, ri, bi, input_size, hidden_size, f_activation);
        auto ft = generate_gate_subgraph(x, h, wf, rf, bf, input_size, hidden_size, f_activation);
        auto ot = generate_gate_subgraph(x, h, wo, ro, bo, input_size, hidden_size, f_activation);
        auto c1t = generate_gate_subgraph(x, h, wc, rc, bc, input_size, hidden_size, g_activation);

        auto it_mul_c1t = std::make_shared<op::v1::Multiply>(it, c1t);
        auto ft_mul_c = std::make_shared<op::v1::Multiply>(ft, c);
        auto ct = std::make_shared<op::v1::Add>(ft_mul_c, it_mul_c1t);

        auto ct_activated = get_activation_function(h_activation, ct);
        auto ht = std::make_shared<op::v1::Multiply>(ct_activated, ot);

        auto c_neg = std::make_shared<op::v0::Negative>(ct);
        auto h_abs = std::make_shared<op::v0::Abs>(ht);

        model = std::make_shared<Model>(NodeVector{h_abs, c_neg}, ParameterVector{x, h, c});
        manager.register_pass<ov::pass::LSTMCellFusion>();
    }

    {
        auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
        auto h = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
        auto c = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});

        // concatenate weights in fico format
        auto w = prepare_weight_fico(wf, wi, wc, wo, ov::Shape{input_size, hidden_size});
        auto r = prepare_weight_fico(rf, ri, rc, ro, ov::Shape{hidden_size, hidden_size});

        std::vector<float> b_value = bf;
        b_value.insert(b_value.end(), bi.begin(), bi.end());
        b_value.insert(b_value.end(), bc.begin(), bc.end());
        b_value.insert(b_value.end(), bo.begin(), bo.end());
        auto b = op::v0::Constant::create(element::f32, Shape{4 * hidden_size}, b_value);

        auto lstm_cell =
            std::make_shared<op::v4::LSTMCell>(x,
                                               h,
                                               c,
                                               w,
                                               r,
                                               b,
                                               hidden_size,
                                               std::vector<std::string>{f_activation, g_activation, h_activation});

        auto c_neg = std::make_shared<op::v0::Negative>(lstm_cell->output(1));
        auto h_abs = std::make_shared<op::v0::Abs>(lstm_cell->output(0));

        model_ref = std::make_shared<Model>(NodeVector{h_abs, c_neg}, ParameterVector{x, h, c});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(LSTMCellFusion,
                         LSTMCellFusionWithSplitWeights,
                         testing::Combine(testing::Values("sigmoid", "tanh", "relu"),
                                          testing::Values("sigmoid", "relu"),
                                          testing::Values("tanh", "relu"),
                                          testing::Values(2, 3),
                                          testing::Values(3, 4)));
