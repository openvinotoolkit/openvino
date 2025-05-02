// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pack_qkv_proj.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::opset8;

TEST_F(TransformationTestsF, FuseThreeMatMulsWithSharedL2Input) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 768});
        auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{1}, {2}));
        auto reduce = std::make_shared<ReduceSum>(pow, Constant::create(element::i64, Shape{1}, {1}), true);
        auto sqrt = std::make_shared<Sqrt>(reduce);
        auto div = std::make_shared<Divide>(input, sqrt);
        auto scale = Constant::create(element::f32, Shape{1}, {1.0f});
        auto norm = std::make_shared<Multiply>(div, scale);

        auto w1 = Constant::create(element::f32, Shape{768, 768}, {0.1f});
        auto w2 = Constant::create(element::f32, Shape{768, 768}, {0.2f});
        auto w3 = Constant::create(element::f32, Shape{768, 768}, {0.3f});

        auto mm1 = std::make_shared<MatMul>(norm, w1);
        auto mm2 = std::make_shared<MatMul>(norm, w2);
        auto mm3 = std::make_shared<MatMul>(norm, w3);

        auto stub = std::make_shared<Relu>(mm1);
        auto stub2 = std::make_shared<Relu>(mm2);
        auto stub3 = std::make_shared<Relu>(mm3);

        model = std::make_shared<Model>(OutputVector{stub, stub2, stub3}, ParameterVector{input});
        manager.register_pass<ov::pass::PackQKVProj>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 768});
        auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{1}, {2}));
        auto reduce = std::make_shared<ReduceSum>(pow, Constant::create(element::i64, Shape{1}, {1}), true);
        auto sqrt = std::make_shared<Sqrt>(reduce);
        auto div = std::make_shared<Divide>(input, sqrt);
        auto scale = Constant::create(element::f32, Shape{1}, {1.0f});
        auto norm = std::make_shared<Multiply>(div, scale);

        auto w1 = Constant::create(element::f32, Shape{768, 768}, {0.1f});
        auto w2 = Constant::create(element::f32, Shape{768, 768}, {0.2f});
        auto w3 = Constant::create(element::f32, Shape{768, 768}, {0.3f});

        // auto packed_weights = std::make_shared<Concat>(OutputVector{w1, w2, w3}, 1);
        auto packed_weights = ov::op::util::make_try_fold<Concat>(OutputVector{w1, w2, w3}, 1);
        auto fused_mm = std::make_shared<MatMul>(norm, packed_weights);

        auto axis = Constant::create(element::i64, Shape{}, {1});
        auto sizes = Constant::create(element::i64, Shape{3}, {768, 768, 768});
        auto split = std::make_shared<VariadicSplit>(fused_mm, axis, sizes);

        auto stub = std::make_shared<Relu>(split->output(0));
        auto stub2 = std::make_shared<Relu>(split->output(1));
        auto stub3 = std::make_shared<Relu>(split->output(2));
        model_ref = std::make_shared<Model>(OutputVector{stub, stub2, stub3}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
