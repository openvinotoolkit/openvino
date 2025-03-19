// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dilated_convolution_converter.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, DilatedConvolutionConverter) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        model = std::make_shared<Model>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto conv = std::make_shared<opset6::Convolution>(data,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          Strides{2, 2},
                                                          op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, NegativeDilatedConvolutionConverterPadsLessThanCrops) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
}

TEST_F(TransformationTestsF, NegativeDilatedConvolutionConverterNonZeroPadsForNC) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 5, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch,
                                                          filters,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        model = std::make_shared<Model>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
}

TEST_F(TransformationTestsF, DilatedGroupConvolutionConverter) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 1, 1, 3, 3});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2}));
        auto conv = std::make_shared<opset6::GroupConvolution>(space_to_batch,
                                                               filters,
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
        auto batch_to_space =
            std::make_shared<opset6::BatchToSpace>(conv,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        model = std::make_shared<Model>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        manager.register_pass<ov::pass::DilatedConvolutionConverter>();
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 1, 1, 3, 3});
        auto conv = std::make_shared<opset6::GroupConvolution>(data,
                                                               filters,
                                                               Strides{1, 1},
                                                               CoordinateDiff{1, 1},
                                                               CoordinateDiff{1, 1},
                                                               Strides{2, 2},
                                                               op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
