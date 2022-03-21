// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST_F(TransformationTestsF, BatchToSpaceFusionTranspose) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 1, -1, 2}),
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
        manager.register_pass<pass::BatchToSpaceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto batch_to_space = std::make_shared<opset6::BatchToSpace>(data,
            op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
            op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
            op::Constant::create(element::i64, Shape{4}, {1, 2, 1, 14}));
        function_ref = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, BatchToSpaceFusionReshape) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{4}, {0, 0, 3, 0}),
                op::Constant::create(element::i64, Shape{4}, {2, 1, 7, -2}),
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});
        auto reshape_after = std::make_shared<opset6::Reshape>(slice, op::Constant::create(element::i64, Shape{4}, {1, 2, 4, 14}), false);
        function = std::make_shared<Function>(NodeVector{reshape_after}, ParameterVector{data});
        manager.register_pass<pass::BatchToSpaceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 3, 4, 8});
        auto batch_to_space = std::make_shared<opset6::BatchToSpace>(data,
            op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
            op::Constant::create(element::i64, Shape{4}, {0, 0, 3, 0}),
            op::Constant::create(element::i64, Shape{4}, {1, 0, 1, 2}));
        function_ref = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeBatchToSpaceFusionInvalidTransposePerm) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {2, 0, 1, 3}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 1, -1, 2}),
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
        manager.register_pass<pass::BatchToSpaceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {2, 0, 1, 3}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 1, -1, 2}),
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeBatchToSpaceFusionInvalidMode) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 1, -1, 2}),
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
        manager.register_pass<pass::BatchToSpaceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 1, -1, 2}),
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeBatchToSpaceFusionInvalidRank) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{16, 3, 4, 8, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{5}, {0, 0, 2, 1, 1}),
                op::Constant::create(element::i64, Shape{5}, {2, 1, -1, 2, 2}),
                std::vector<int64_t>{0, 0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
        manager.register_pass<pass::BatchToSpaceFusion>();
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{16, 3, 4, 8, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto depth_to_space = std::make_shared<opset6::DepthToSpace>(trans_before, opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        auto slice = std::make_shared<opset6::StridedSlice>(depth_to_space,
                op::Constant::create(element::i64, Shape{5}, {0, 0, 2, 1, 1}),
                op::Constant::create(element::i64, Shape{5}, {2, 1, -1, 2, 2}),
                std::vector<int64_t>{0, 0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0, 0});
        auto trans_after = std::make_shared<opset6::Transpose>(slice, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
        manager.register_pass<pass::BatchToSpaceFusion>();
    }
}

