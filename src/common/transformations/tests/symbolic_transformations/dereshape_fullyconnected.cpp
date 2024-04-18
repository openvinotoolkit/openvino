// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "transformations/symbolic_transformations/dereshape_matmul.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

TEST_F(TransformationTestsF, DeReshapeFC) {
    auto shape = PartialShape{-1, -1, 40};
    set_shape_symbols(shape);  // we label shape with consecutive labels: A, B, C

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto in_reshape = make_shared<v1::Reshape>(data, v0::Constant::create(element::i64, {2}, {-1, 40}), true);
        auto second_input = make_shared<v0::Parameter>(element::f32, Shape{40, 80});

        auto matmul = make_shared<v0::MatMul>(in_reshape, second_input);

        auto batch_dims = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {0, 1});
        auto pattern =
            make_shared<v0::Concat>(OutputVector{batch_dims, v0::Constant::create(element::i64, {1}, {80})}, 0);
        auto out_reshape = make_shared<v1::Reshape>(matmul, pattern, false);

        model = make_shared<Model>(NodeVector{out_reshape}, ParameterVector{data, second_input});
        manager.register_pass<pass::DeReshapeFullyConnected>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto second_input = make_shared<v0::Parameter>(element::f32, Shape{40, 80});
        auto matmul = make_shared<v0::MatMul>(data, second_input);

        model_ref = make_shared<Model>(NodeVector{matmul}, ParameterVector{data, second_input});
    }
}

TEST_F(TransformationTestsF, DeReshapeFCWithConvert) {
    auto shape = PartialShape{-1, -1, 40};
    set_shape_symbols(shape);  // we label shape with consecutive labels: A, B, C
    {
        auto data = make_shared<v0::Parameter>(element::f16, shape);
        auto in_reshape = make_shared<v1::Reshape>(data, v0::Constant::create(element::i64, {2}, {-1, 40}), true);
        auto convert = make_shared<v0::Convert>(in_reshape, element::f32);
        auto second_input = make_shared<v0::Parameter>(element::f32, Shape{40, 80});

        auto matmul = make_shared<v0::MatMul>(convert, second_input);

        auto batch_dims = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {0, 1});
        auto pattern =
            make_shared<v0::Concat>(OutputVector{batch_dims, v0::Constant::create(element::i64, {1}, {80})}, 0);
        auto out_reshape = make_shared<v1::Reshape>(matmul, pattern, false);

        model = make_shared<Model>(NodeVector{out_reshape}, ParameterVector{data, second_input});
        manager.register_pass<pass::DeReshapeFullyConnected>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f16, shape);
        auto convert = make_shared<v0::Convert>(data, element::f32);
        auto second_input = make_shared<v0::Parameter>(element::f32, Shape{40, 80});
        auto matmul = make_shared<v0::MatMul>(convert, second_input);

        model_ref = make_shared<Model>(NodeVector{matmul}, ParameterVector{data, second_input});
    }
}

TEST_F(TransformationTestsF, DeReshapeFCNegative) {
    auto shape = PartialShape{-1, -1, 40};
    set_shape_symbols(shape);  // we label shape with consecutive labels: A, B, C
    {
        auto data = make_shared<v0::Parameter>(element::f16, shape);
        auto in_reshape = make_shared<v1::Reshape>(data, v0::Constant::create(element::i64, {2}, {-1, 40}), true);
        auto convert = make_shared<v0::Convert>(in_reshape, element::f32);
        auto second_input = make_shared<v0::Parameter>(element::f32, Shape{40, 80});

        auto matmul = make_shared<v0::MatMul>(convert, second_input);

        auto pattern = v0::Constant::create(element::i64, {3}, {4, -1, 80});
        auto out_reshape = make_shared<v1::Reshape>(matmul, pattern, false);

        model = make_shared<Model>(NodeVector{out_reshape}, ParameterVector{data, second_input});
        manager.register_pass<pass::DeReshapeFullyConnected>();
    }
}