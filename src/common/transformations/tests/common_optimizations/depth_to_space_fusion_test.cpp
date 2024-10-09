// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/depth_to_space_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, DepthToSpaceFusionDepthFirst) {
    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto shape_reshape_before = opset3::Constant::create(element::i64, Shape{6}, {1, 32, 2, 2, 720, 480});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 1, 4, 2, 5, 3});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);

        model = std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::as_type_ptr<const opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ov::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto depth_to_space =
            std::make_shared<opset3::DepthToSpace>(input0, opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        model_ref = std::make_shared<ov::Model>(NodeVector{depth_to_space}, ParameterVector{input0});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionDepthFirstDynamicBatch) {
    {
        auto input_pshape = PartialShape{Dimension::dynamic(), 128, 720, 480};
        auto input = std::make_shared<opset3::Parameter>(element::f32, input_pshape);
        auto shape_reshape_before = opset3::Constant::create(element::i64, Shape{6}, {1, 32, 2, 2, 720, 480});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 1, 4, 2, 5, 3});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);

        model = std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{input});

        auto callback = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::as_type_ptr<const opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ov::pass::DepthToSpaceFusion>();
    }

    {
        auto input_pshape = PartialShape{Dimension::dynamic(), 128, 720, 480};
        auto input = std::make_shared<opset3::Parameter>(element::f32, input_pshape);
        auto depth_to_space =
            std::make_shared<opset3::DepthToSpace>(input, opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        model_ref = std::make_shared<ov::Model>(NodeVector{depth_to_space}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionBlockFirst) {
    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto shape_reshape_before = opset3::Constant::create(element::i64, Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);

        model = std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::as_type_ptr<const opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ov::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto depth_to_space =
            std::make_shared<opset3::DepthToSpace>(input0, opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        model_ref = std::make_shared<ov::Model>(NodeVector{depth_to_space}, ParameterVector{input0});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionBlockFirstDynamicBatch) {
    {
        auto input_pshape = PartialShape{Dimension::dynamic(), 128, 720, 480};
        auto input = std::make_shared<opset3::Parameter>(element::f32, input_pshape);
        auto shape_reshape_before = opset3::Constant::create(element::i64, Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);

        model = std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{input});

        auto callback = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::as_type_ptr<const opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ov::pass::DepthToSpaceFusion>();
    }

    {
        auto input_pshape = PartialShape{Dimension::dynamic(), 128, 720, 480};
        auto input = std::make_shared<opset3::Parameter>(element::f32, input_pshape);
        auto depth_to_space =
            std::make_shared<opset3::DepthToSpace>(input, opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        model_ref = std::make_shared<ov::Model>(NodeVector{depth_to_space}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionDynamicShape) {
    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto shape_reshape_before = std::make_shared<opset3::Parameter>(element::i64, Shape{6});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);

        model = std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{input0, shape_reshape_before});

        auto callback = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::as_type_ptr<const opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ov::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto shape_reshape_before = std::make_shared<opset3::Parameter>(element::i64, Shape{6});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);

        model_ref =
            std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{input0, shape_reshape_before});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionSeveralConsumers) {
    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto shape_reshape_before = opset3::Constant::create(element::i64, Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);
        auto result = std::make_shared<opset3::Result>(reshape_after);

        // additional consumer
        auto additional_consumer = std::make_shared<opset3::Result>(reshape_before);

        model = std::make_shared<ov::Model>(NodeVector{result, additional_consumer}, ParameterVector{input0});
        auto callback = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::as_type_ptr<const opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ov::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 720, 480});
        auto shape_reshape_before = opset3::Constant::create(element::i64, Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = opset3::Constant::create(element::i64, Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = opset3::Constant::create(element::i64, Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, shape_reshape_after, false);
        auto result = std::make_shared<opset3::Result>(reshape_after);

        // additional consumer
        auto additional_consumer = std::make_shared<opset3::Result>(reshape_before);

        model_ref = std::make_shared<ov::Model>(NodeVector{result, additional_consumer}, ParameterVector{input0});
    }
}
