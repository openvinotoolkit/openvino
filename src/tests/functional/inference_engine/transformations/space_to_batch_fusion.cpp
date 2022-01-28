// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, SpaceToBatchFusionTranspose) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
            op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
            op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
            op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        function_ref = std::make_shared<Function>(NodeVector{space_to_batch}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchFusionReshape) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto reshape_before = std::make_shared<opset6::Reshape>(data, op::Constant::create(element::i64, Shape{4}, {3, 12, 4, 8}), false);
        auto pad = std::make_shared<opset6::Pad>(reshape_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
            op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
            op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
            op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        function_ref = std::make_shared<Function>(NodeVector{space_to_batch}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidTransposePerm) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidPad) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {1}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {1}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidMode) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidRank) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{5}, {0, 1, 1, 0, 0}),
                op::Constant::create(element::i64, Shape{5}, {1, 1, 3, 2, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        function = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{5}, {0, 1, 1, 0, 0}),
                op::Constant::create(element::i64, Shape{5}, {1, 1, 3, 2, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        function_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }
}

