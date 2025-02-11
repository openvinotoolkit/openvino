// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gru_cell_fusion.hpp"

#include <gtest/gtest.h>

#include <queue>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset9;
using namespace ov::element;
using namespace testing;

namespace {

enum class WeightsFormat { zr, rz };

Output<Node> create_activation_by_name(const std::string& activation_name, const Output<Node>& input) {
    if (activation_name == "sigmoid") {
        return make_shared<Sigmoid>(input);
    } else if (activation_name == "tanh") {
        return make_shared<Tanh>(input);
    } else if (activation_name == "relu") {
        return make_shared<Relu>(input);
    }
    OPENVINO_ASSERT(false, "Unsupported activation function");
}

shared_ptr<Model> gen_model(WeightsFormat format,
                            const std::string& activation_1,
                            const std::string& activation_2,
                            size_t batch,
                            size_t hidden_size,
                            size_t input_size,
                            bool use_bias_add_1,
                            bool use_bias_add_2,
                            bool use_dyn_shapes) {
    int r_idx = 0, z_idx = 1;
    if (format == WeightsFormat::zr) {
        swap(r_idx, z_idx);
    }
    auto X = make_shared<Parameter>(f32, Shape{batch, input_size});
    if (use_dyn_shapes) {
        X = make_shared<Parameter>(f32, PartialShape{static_cast<int64_t>(batch), Dimension::dynamic()});
    }
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto WRzr = make_shared<Parameter>(f32, Shape{2 * hidden_size, input_size + hidden_size});
    auto Bzr = make_shared<Parameter>(f32, Shape{1, 2 * hidden_size});
    auto WRh = make_shared<Parameter>(f32, Shape{hidden_size, input_size + hidden_size});
    auto Bh = make_shared<Parameter>(f32, Shape{1, hidden_size});
    auto concat_1 = make_shared<Concat>(OutputVector{X, H}, 1);
    auto matmul_1 = make_shared<MatMul>(concat_1, WRzr, false, true);
    Output<Node> in_to_activation_1 = matmul_1;
    if (use_bias_add_1) {
        in_to_activation_1 = make_shared<Add>(matmul_1, Bzr);
    }

    auto act_1 = create_activation_by_name(activation_1, in_to_activation_1);
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto split = make_shared<Split>(act_1, axis_1, 2);

    auto multiply_1 = make_shared<Multiply>(split->output(r_idx), H);
    auto concat_2 = make_shared<Concat>(OutputVector{X, multiply_1}, 1);
    auto matmul_2 = make_shared<MatMul>(concat_2, WRh, false, true);
    Output<Node> in_to_activation_2 = matmul_2;
    if (use_bias_add_2) {
        in_to_activation_2 = make_shared<Add>(matmul_2, Bh);
    }

    auto act_2 = create_activation_by_name(activation_2, in_to_activation_2);
    auto one = make_shared<Constant>(f32, Shape{1}, 1);
    auto subtract = make_shared<Subtract>(one, split->output(z_idx));
    auto multiply_2 = make_shared<Multiply>(subtract, act_2);
    auto multiply_3 = make_shared<Multiply>(split->output(z_idx), H);
    auto add = make_shared<Add>(multiply_2, multiply_3);
    return make_shared<Model>(OutputVector{add}, ParameterVector{X, H, WRzr, WRh, Bzr, Bh});
}

shared_ptr<Model> gen_reference(WeightsFormat format,
                                const std::string& activation_1,
                                const std::string& activation_2,
                                size_t batch,
                                size_t hidden_size,
                                size_t input_size,
                                bool use_bias_add_1,
                                bool use_bias_add_2) {
    auto X = make_shared<Parameter>(f32, Shape{batch, input_size});
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto WR = make_shared<Parameter>(f32, Shape{2 * hidden_size, input_size + hidden_size});
    auto WRh = make_shared<Parameter>(f32, Shape{hidden_size, input_size + hidden_size});
    ParameterVector params = {X, H, WR, WRh};

    shared_ptr<Node> B = make_shared<Constant>(f32, Shape{1, 2 * hidden_size}, 0);
    if (use_bias_add_1) {
        B = make_shared<Parameter>(f32, Shape{1, 2 * hidden_size});
        params.push_back(ov::as_type_ptr<Parameter>(B));
    }
    shared_ptr<Node> Bh = make_shared<Constant>(f32, Shape{1, hidden_size}, 0);
    if (use_bias_add_2) {
        Bh = make_shared<Parameter>(f32, Shape{1, hidden_size});
        params.push_back(ov::as_type_ptr<Parameter>(Bh));
    }

    auto axis_0 = make_shared<Constant>(i64, Shape{}, 0);
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto split_lenghts = make_shared<Constant>(i64, Shape{2}, vector<size_t>{input_size, hidden_size});
    auto split_WRh = make_shared<VariadicSplit>(WRh, axis_1, split_lenghts);

    Output<Node> Wzrh, Rzrh, Bzrh;
    if (format == WeightsFormat::zr) {
        auto split_WRzr = make_shared<VariadicSplit>(WR, axis_1, split_lenghts);
        Wzrh = make_shared<Concat>(OutputVector{split_WRzr->output(0), split_WRh->output(0)}, 0);
        Rzrh = make_shared<Concat>(OutputVector{split_WRzr->output(1), split_WRh->output(1)}, 0);
        Bzrh = make_shared<Concat>(OutputVector{B, Bh}, 1);
    } else {
        auto split_WRrz = make_shared<VariadicSplit>(WR, axis_1, split_lenghts);
        auto split_W_r_z = make_shared<Split>(split_WRrz->output(0), axis_0, 2);
        auto split_R_r_z = make_shared<Split>(split_WRrz->output(1), axis_0, 2);
        Wzrh =
            make_shared<Concat>(OutputVector{split_W_r_z->output(1), split_W_r_z->output(0), split_WRh->output(0)}, 0);
        Rzrh =
            make_shared<Concat>(OutputVector{split_R_r_z->output(1), split_R_r_z->output(0), split_WRh->output(1)}, 0);
        auto split_B_r_z = make_shared<Split>(B, axis_1, 2);
        Bzrh = make_shared<Concat>(OutputVector{split_B_r_z->output(1), split_B_r_z->output(0), Bh}, 1);
    }

    auto squeeze_B = make_shared<Squeeze>(Bzrh, axis_0);
    auto cell =
        make_shared<GRUCell>(X, H, Wzrh, Rzrh, squeeze_B, hidden_size, vector<std::string>{activation_1, activation_2});
    return make_shared<Model>(OutputVector{cell}, params);
}
}  // namespace

struct GRUFusionParams {
    WeightsFormat format;
    std::string activation_1;
    std::string activation_2;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    bool use_bias_add_1;
    bool use_bias_add_2;
};

class GRUFusionTest : public WithParamInterface<GRUFusionParams>, public TransformationTestsF {};

TEST_P(GRUFusionTest, GRUCellPattern) {
    const auto& p = GetParam();
    {
        model = gen_model(p.format,
                          p.activation_1,
                          p.activation_2,
                          p.batch,
                          p.hidden_size,
                          p.input_size,
                          p.use_bias_add_1,
                          p.use_bias_add_2,
                          false);
        manager.register_pass<ov::pass::GRUCellFusion>();
    }

    {
        model_ref = gen_reference(p.format,
                                  p.activation_1,
                                  p.activation_2,
                                  p.batch,
                                  p.hidden_size,
                                  p.input_size,
                                  p.use_bias_add_1,
                                  p.use_bias_add_2);
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

class GRUFusionTestDyn : public WithParamInterface<GRUFusionParams>, public TransformationTestsF {};

TEST_P(GRUFusionTestDyn, GRUCellPatternDynamicShapes) {
    const auto& p = GetParam();
    {
        model = gen_model(p.format,
                          p.activation_1,
                          p.activation_2,
                          p.batch,
                          p.hidden_size,
                          p.input_size,
                          false,
                          false,
                          true);
        manager.register_pass<ov::pass::GRUCellFusion>();  // the transformation won't be applied
    }
}

static const vector<GRUFusionParams> params = {
    GRUFusionParams{WeightsFormat::zr, "sigmoid", "tanh", 1, 1, 1, true, true},
    GRUFusionParams{WeightsFormat::zr, "tanh", "sigmoid", 2, 128, 32, true, true},
    GRUFusionParams{WeightsFormat::zr, "tanh", "tanh", 2, 128, 32, true, false},
    GRUFusionParams{WeightsFormat::zr, "sigmoid", "relu", 2, 128, 32, false, false},
    GRUFusionParams{WeightsFormat::zr, "relu", "tanh", 2, 128, 32, false, true},
    GRUFusionParams{WeightsFormat::rz, "sigmoid", "tanh", 1, 1, 1, true, true},
    GRUFusionParams{WeightsFormat::rz, "tanh", "sigmoid", 2, 128, 32, true, true},
    GRUFusionParams{WeightsFormat::rz, "relu", "sigmoid", 2, 128, 32, true, true},
    GRUFusionParams{WeightsFormat::rz, "tanh", "tanh", 2, 128, 32, true, false},
    GRUFusionParams{WeightsFormat::rz, "sigmoid", "relu", 2, 128, 32, false, false},
    GRUFusionParams{WeightsFormat::rz, "relu", "tanh", 2, 128, 32, false, true},
};

INSTANTIATE_TEST_SUITE_P(GRUFusionTest, GRUFusionTest, ValuesIn(params));
INSTANTIATE_TEST_SUITE_P(GRUFusionTestDyn, GRUFusionTestDyn, ValuesIn(params));
