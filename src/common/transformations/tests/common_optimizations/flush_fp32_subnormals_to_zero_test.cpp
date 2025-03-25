// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/flush_fp32_subnormals_to_zero.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset10;
using namespace std;

namespace {
union FloatIntUnion {
    uint32_t u;
    float f;
};
FloatIntUnion maximum_subnorm_val = {0x007fffff};  // = 2^−126 * (1 - 2^−23) ~= 1.1754942107e-38f
FloatIntUnion minimum_subnorm_val = {0x00000001};  // = 2^−149 ~= 1.4012984643e-45f
FloatIntUnion minimum_norm_val = {0x00800000};     // = 2^−126 ~= 1.1754943508-38f
}  // namespace

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_max_subnorm) {
    float subnormal_val = maximum_subnorm_val.f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_min_subnorm) {
    float subnormal_val = minimum_subnorm_val.f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_arbitrary_subnorm) {
    float subnormal_val = 2.0e-44f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_max_neg_subnorm) {
    float subnormal_val = -maximum_subnorm_val.f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_min_neg_subnorm) {
    float subnormal_val = -minimum_subnorm_val.f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_arbitrary_neg_subnorm) {
    float subnormal_val = -2.0e-45f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val,
                                               subnormal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_arbitrary_norm) {
    // minimum normalized val should not be flushed to zero
    float normal_val = minimum_norm_val.f;
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               normal_val,
                                               normal_val,
                                               normal_val,
                                               normal_val,
                                               normal_val,
                                               normal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        manager.register_pass<pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 12, 12});

        auto const_weights = Constant::create(element::f32,
                                              Shape{1, 3, 4, 1},
                                              {0.0f,
                                               1.0f,
                                               2.0f,
                                               3.0f,
                                               4.0f,
                                               5.0f,
                                               normal_val,
                                               normal_val,
                                               normal_val,
                                               normal_val,
                                               normal_val,
                                               normal_val});
        auto conv = std::make_shared<Convolution>(input,
                                                  const_weights,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
