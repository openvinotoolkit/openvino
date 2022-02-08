// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <sstream>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST_F(TransformationTestsF, DepthToSpaceFusionDepthFirst) {
    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 32, 2, 2, 720, 480});
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 1, 4, 2, 5, 3});
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape> (permute, shape_reshape_after, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
        };


        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto depth_to_space = std::make_shared<ngraph::opset3::DepthToSpace>(input0, ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionDepthFirstDynamicBatch) {
    {
        auto input_pshape = ngraph::PartialShape{ ngraph::Dimension::dynamic(), 128, 720, 480 };
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, input_pshape);
        auto shape_reshape_before = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 1, 32, 2, 2, 720, 480 });
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 1, 4, 2, 5, 3 });
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 1, 32, 1440, 960 });

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(input, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, shape_reshape_after, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    }

    {
        auto input_pshape = ngraph::PartialShape{ ngraph::Dimension::dynamic(), 128, 720, 480 };
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, input_pshape);
        auto depth_to_space = std::make_shared<ngraph::opset3::DepthToSpace>(input, ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ depth_to_space }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionBlockFirst) {
    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape> (permute, shape_reshape_after, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
        };


        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto depth_to_space = std::make_shared<ngraph::opset3::DepthToSpace>(input0, ngraph::opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionBlockFirstDynamicBatch) {
    {
        auto input_pshape = ngraph::PartialShape{ ngraph::Dimension::dynamic(), 128, 720, 480 };
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, input_pshape);
        auto shape_reshape_before = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 1, 2, 2, 32, 720, 480 });
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 3, 4, 1, 5, 2 });
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 1, 32, 1440, 960 });

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(input, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, shape_reshape_after, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    }

    {
        auto input_pshape = ngraph::PartialShape{ ngraph::Dimension::dynamic(), 128, 720, 480 };
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, input_pshape);
        auto depth_to_space = std::make_shared<ngraph::opset3::DepthToSpace>(input, ngraph::opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ depth_to_space }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionDynamicShape) {
    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{6});
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape> (permute, shape_reshape_after, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0, shape_reshape_before});

        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
        };


        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{6});
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape> (permute, shape_reshape_after, false);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0, shape_reshape_before});
    }
}

TEST_F(TransformationTestsF, DepthToSpaceFusionSeveralConsumers) {
    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape> (permute, shape_reshape_after, false);
        auto result = std::make_shared<ngraph::opset3::Result>(reshape_after);

        // additional consumer
        auto additional_consumer = std::make_shared<ngraph::opset3::Result> (reshape_before);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ result, additional_consumer }, ngraph::ParameterVector{ input0 });
        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
        };

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    }

    {
        auto input0 = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 128, 720, 480 });
        auto shape_reshape_before = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 1, 2, 2, 32, 720, 480 });
        auto permutation = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 3, 4, 1, 5, 2 });
        auto shape_reshape_after = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 1, 32, 1440, 960 });

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, shape_reshape_after, false);
        auto result = std::make_shared<ngraph::opset3::Result>(reshape_after);

        // additional consumer
        auto additional_consumer = std::make_shared<ngraph::opset3::Result>(reshape_before);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ result, additional_consumer }, ngraph::ParameterVector{ input0 });
    }
}
