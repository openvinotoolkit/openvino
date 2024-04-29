// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/nop_broadcast.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

TEST_F(TransformationTestsF, NopBroadcastOpset1) {
    auto shape = PartialShape::dynamic(4);
    set_shape_symbols(shape);  // we set unique symbols to the shape: A, B, C, D

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto symbol_input = make_shared<v0::Parameter>(element::f32, shape);
        auto shape_of = make_shared<v3::ShapeOf>(symbol_input);
        auto ones = ov::op::v0::Constant::create(element::i64, {}, {1});
        auto maximum = make_shared<v1::Maximum>(shape_of, ones);

        auto broadcast = make_shared<v1::Broadcast>(data, maximum);
        auto relu = make_shared<v0::Relu>(broadcast);

        model = make_shared<Model>(NodeVector{relu}, ParameterVector{data, symbol_input});
        manager.register_pass<pass::NopBroadcast>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto relu = make_shared<v0::Relu>(data);

        auto symbol_input = make_shared<v0::Parameter>(element::f32, shape);

        model_ref = make_shared<Model>(NodeVector{relu}, ParameterVector{data, symbol_input});
    }
}

TEST_F(TransformationTestsF, NopBroadcastOpset3) {
    auto shape = PartialShape::dynamic(4);
    set_shape_symbols(shape);  // we set unique symbols to the shape: A, B, C, D

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto symbol_input = make_shared<v0::Parameter>(element::f32, shape);
        auto shape_of = make_shared<v3::ShapeOf>(symbol_input);
        auto ones = ov::op::v0::Constant::create(element::i64, {4}, {1, 1, 1, 1});
        auto maximum = make_shared<v1::Maximum>(shape_of, ones);

        auto broadcast = make_shared<v3::Broadcast>(data, maximum);
        auto relu = make_shared<v0::Relu>(broadcast);

        model = make_shared<Model>(NodeVector{relu}, ParameterVector{data, symbol_input});
        manager.register_pass<pass::NopBroadcast>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto relu = make_shared<v0::Relu>(data);

        auto symbol_input = make_shared<v0::Parameter>(element::f32, shape);

        model_ref = make_shared<Model>(NodeVector{relu}, ParameterVector{data, symbol_input});
    }
}

TEST_F(TransformationTestsF, NopBroadcastNegative) {
    {
        auto shape = PartialShape::dynamic(1);
        shape[0].set_symbol(std::make_shared<ov::Symbol>());

        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto symbol_input = make_shared<v0::Parameter>(element::f32, shape);
        auto shape_of = make_shared<v3::ShapeOf>(symbol_input);
        auto ones = ov::op::v0::Constant::create(element::i64, {2}, {1, 1});
        auto maximum = make_shared<v1::Maximum>(shape_of, ones);

        auto broadcast = make_shared<v1::Broadcast>(data, maximum);
        auto relu = make_shared<v0::Relu>(broadcast);

        model = make_shared<Model>(NodeVector{relu}, ParameterVector{data, symbol_input});
        manager.register_pass<pass::NopBroadcast>();
    }
}
