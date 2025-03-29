// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/space_to_batch_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, SpaceToBatchFusionTranspose) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        model_ref = std::make_shared<Model>(NodeVector{space_to_batch}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchFusionTransposePad12) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<op::v12::Pad>(trans_before,
                                                  op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                  op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                                                  op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                  op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        model_ref = std::make_shared<Model>(NodeVector{space_to_batch}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchFusionTransposeNegativePads) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<op::v12::Pad>(trans_before,
                                                  op::v0::Constant::create(element::i64, Shape{4}, {1, 1, -1, -1}),
                                                  op::v0::Constant::create(element::i64, Shape{4}, {2, 2, -3, -3}),
                                                  op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                  op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 4);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }
}

TEST_F(TransformationTestsF, SpaceToBatchFusionReshape) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto reshape_before =
            std::make_shared<opset6::Reshape>(data,
                                              op::v0::Constant::create(element::i64, Shape{4}, {3, 12, 4, 8}),
                                              false);
        auto pad = std::make_shared<opset6::Pad>(reshape_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch =
            std::make_shared<opset6::SpaceToBatch>(data,
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                   op::v0::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        model_ref = std::make_shared<Model>(NodeVector{space_to_batch}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidTransposePerm) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model_ref = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidPad) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {1}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {1}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model_ref = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidMode) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data, op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                                                 op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        model_ref = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeSpaceToBatchFusionInvalidRank) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data,
                                                op::v0::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{5}, {0, 1, 1, 0, 0}),
                                                 op::v0::Constant::create(element::i64, Shape{5}, {1, 1, 3, 2, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});

        manager.register_pass<ov::pass::SpaceToBatchFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8, 8});
        auto trans_before =
            std::make_shared<opset6::Transpose>(data,
                                                op::v0::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                                                 op::v0::Constant::create(element::i64, Shape{5}, {0, 1, 1, 0, 0}),
                                                 op::v0::Constant::create(element::i64, Shape{5}, {1, 1, 3, 2, 2}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}),
                                                 op::PadMode::CONSTANT);
        auto space_to_depth =
            std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after =
            std::make_shared<opset6::Transpose>(space_to_depth,
                                                op::v0::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        model_ref = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{data});
    }
}
