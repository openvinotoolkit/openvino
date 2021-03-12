// Copyright (C) 2021 Intel Corporation
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

TEST(TransformationTests, SpaceToBatchFusionTranspose) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToBatchFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
            op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
            op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
            op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        f_ref = std::make_shared<Function>(NodeVector{space_to_batch}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToBatchFusionReshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto reshape_before = std::make_shared<opset6::Reshape>(data, op::Constant::create(element::i64, Shape{4}, {3, 12, 4, 8}), false);
        auto pad = std::make_shared<opset6::Pad>(reshape_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToBatchFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
            op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
            op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
            op::Constant::create(element::i64, Shape{4}, {2, 2, 3, 3}));

        f_ref = std::make_shared<Function>(NodeVector{space_to_batch}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeSpaceToBatchFusionInvalidTransposePerm) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {3, 0, 2, 1}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToBatchFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
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
        f_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeSpaceToBatchFusionInvalidPad) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {1}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToBatchFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
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
        f_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeSpaceToBatchFusionInvalidMode) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{4}, {0, 1, 1, 0}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 3, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToBatchFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
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
        f_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeSpaceToBatchFusionInvalidRank) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8, 8});
        auto trans_before = std::make_shared<opset6::Transpose>(data, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        auto pad = std::make_shared<opset6::Pad>(trans_before,
                op::Constant::create(element::i64, Shape{5}, {0, 1, 1, 0, 0}),
                op::Constant::create(element::i64, Shape{5}, {1, 1, 3, 2, 2}),
                op::Constant::create(element::f32, Shape{}, {0}), op::PadMode::CONSTANT);
        auto space_to_depth = std::make_shared<opset6::SpaceToDepth>(pad, opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        auto trans_after = std::make_shared<opset6::Transpose>(space_to_depth, op::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4}));
        f = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToBatchFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
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
        f_ref = std::make_shared<Function>(NodeVector{trans_after}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

