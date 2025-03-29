// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/augru_cell_fusion.hpp"

#include <gtest/gtest.h>

#include <queue>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset9.hpp"
#include "ov_ops/augru_cell.hpp"

using namespace std;
using namespace testing;

using namespace ov;
using namespace opset9;
using namespace element;

namespace {
shared_ptr<Model> gen_model(size_t batch, size_t hidden_size, size_t input_size, bool use_dyn_shapes) {
    auto X = make_shared<Parameter>(f32, Shape{batch, input_size});
    if (use_dyn_shapes) {
        X = make_shared<Parameter>(f32, PartialShape{static_cast<int64_t>(batch), Dimension::dynamic()});
    }
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto WRzr = make_shared<Parameter>(f32, Shape{2 * hidden_size, input_size + hidden_size});
    auto Bzr = make_shared<Parameter>(f32, Shape{1, 2 * hidden_size});
    auto WRh = make_shared<Parameter>(f32, Shape{hidden_size, input_size + hidden_size});
    auto Bh = make_shared<Parameter>(f32, Shape{1, hidden_size});
    auto A = make_shared<Parameter>(f32, Shape{batch, 1});
    auto concat_1 = make_shared<Concat>(OutputVector{X, H}, 1);
    auto matmul_1 = make_shared<MatMul>(concat_1, WRzr, false, true);
    auto in_to_activation_1 = make_shared<Add>(matmul_1, Bzr);

    auto sigmoid = make_shared<Sigmoid>(in_to_activation_1);
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto split = make_shared<Split>(sigmoid, axis_1, 2);

    auto multiply_1 = make_shared<Multiply>(split, H);
    auto concat_2 = make_shared<Concat>(OutputVector{X, multiply_1}, 1);
    auto matmul_2 = make_shared<MatMul>(concat_2, WRh, false, true);
    auto in_to_activation_2 = make_shared<Add>(matmul_2, Bh);
    auto tanh = make_shared<Tanh>(in_to_activation_2);

    auto one = make_shared<Constant>(f32, Shape{1}, 1);
    auto subtract_1 = make_shared<Subtract>(one, A);
    auto multiply_2 = make_shared<Multiply>(subtract_1, split->output(1));
    auto subtract_2 = make_shared<Subtract>(one, multiply_2);
    auto multiply_3 = make_shared<Multiply>(subtract_2, tanh);

    auto multiply_4 = make_shared<Multiply>(multiply_2, H);
    auto add = make_shared<Add>(multiply_4, multiply_3);
    return make_shared<Model>(OutputVector{add}, ParameterVector{X, H, WRzr, WRh, Bzr, Bh, A});
}

shared_ptr<Model> gen_reference(size_t batch, size_t hidden_size, size_t input_size) {
    auto X = make_shared<Parameter>(f32, Shape{batch, input_size});
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto WRrz = make_shared<Parameter>(f32, Shape{2 * hidden_size, input_size + hidden_size});
    auto WRh = make_shared<Parameter>(f32, Shape{hidden_size, input_size + hidden_size});
    auto Brz = make_shared<Parameter>(f32, Shape{1, 2 * hidden_size});
    auto Bh = make_shared<Parameter>(f32, Shape{1, hidden_size});
    auto A = make_shared<Parameter>(f32, Shape{batch, 1});
    ParameterVector params = {X, H, WRrz, WRh, Brz, Bh, A};

    auto axis_0 = make_shared<Constant>(i64, Shape{}, 0);
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto split_lenghts = make_shared<Constant>(i64, Shape{2}, vector<size_t>{input_size, hidden_size});
    auto split_WRrz = make_shared<VariadicSplit>(WRrz, axis_1, split_lenghts);
    auto split_W_r_z = make_shared<Split>(split_WRrz->output(0), axis_0, 2);
    auto split_R_r_z = make_shared<Split>(split_WRrz->output(1), axis_0, 2);
    auto split_WRh = make_shared<VariadicSplit>(WRh, axis_1, split_lenghts);
    auto Wzrh =
        make_shared<Concat>(OutputVector{split_W_r_z->output(1), split_W_r_z->output(0), split_WRh->output(0)}, 0);
    auto Rzrh =
        make_shared<Concat>(OutputVector{split_R_r_z->output(1), split_R_r_z->output(0), split_WRh->output(1)}, 0);

    auto split_bias_r_z = make_shared<Split>(Brz, axis_1, 2);
    auto B = make_shared<Concat>(OutputVector{split_bias_r_z->output(1), split_bias_r_z->output(0), Bh}, 1);

    auto squeeze_B = make_shared<Squeeze>(B, axis_0);
    auto cell = make_shared<ov::op::internal::AUGRUCell>(X, H, Wzrh, Rzrh, squeeze_B, A, hidden_size);
    return make_shared<Model>(OutputVector{cell}, params);
}
}  // namespace

struct AUGRUFusionParams {
    size_t batch;
    size_t hidden_size;
    size_t input_size;
};

class AUGRUFusionTest : public WithParamInterface<AUGRUFusionParams>, public TransformationTestsF {};

TEST_P(AUGRUFusionTest, AUGRUCellPattern) {
    const auto& p = GetParam();
    {
        model = gen_model(p.batch, p.hidden_size, p.input_size, false);
        manager.register_pass<ov::pass::AUGRUCellFusion>();
    }

    { model_ref = gen_reference(p.batch, p.hidden_size, p.input_size); }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

class AUGRUFusionTestDyn : public WithParamInterface<AUGRUFusionParams>, public TransformationTestsF {};

TEST_P(AUGRUFusionTestDyn, AUGRUCellPatternDynamicShapes) {
    const auto& p = GetParam();
    {
        model = gen_model(p.batch, p.hidden_size, p.input_size, true);
        // the transformation won't be applied because we can't determine hidden_size/input_size,
        // they are dynamic.
        manager.register_pass<ov::pass::AUGRUCellFusion>();
    }
}

static const std::vector<AUGRUFusionParams> params = {
    AUGRUFusionParams{1, 1, 1},
    AUGRUFusionParams{2, 128, 32},
};

INSTANTIATE_TEST_SUITE_P(AUGRUFusionTest, AUGRUFusionTest, ValuesIn(params));
INSTANTIATE_TEST_SUITE_P(AUGRUFusionTestDyn, AUGRUFusionTestDyn, ValuesIn(params));
