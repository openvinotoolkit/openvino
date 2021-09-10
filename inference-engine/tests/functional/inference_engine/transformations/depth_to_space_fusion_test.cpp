// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST(TransformationTests, DepthToSpaceFusionDepthFirst) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 32, 2, 2, 720, 480});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 1, 4, 2, 5, 3});
        auto shape_reshape_after =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto depth_to_space =
            std::make_shared<ngraph::opset8::DepthToSpace>(input0,
                                                           ngraph::opset8::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
                                                           2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionBlockFirst) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto depth_to_space =
            std::make_shared<ngraph::opset8::DepthToSpace>(input0,
                                                           ngraph::opset8::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                           2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, DepthToSpaceFusionDepthFirstWithMoreEntries) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 80, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 360, 640, 20, 2, 2});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 5, 2, 3, 1, 4});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 640, 20, 720});
        auto transpose2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 2, 1, 3, 4});
        auto reshape2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1280, 20, 720});
        auto transpose3_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(input0, transpose0_intput1);
        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape0_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose1_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape1_intput1, false);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose2_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose2, reshape2_intput1, false);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose3_intput1);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose3}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 80, 360, 640});
        auto depth_to_space =
            std::make_shared<ngraph::opset8::DepthToSpace>(input0,
                                                           ngraph::opset8::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
                                                           2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionBlockFirstWithMoreEntries) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{5, 270, 360, 640});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {5, 3, 3, 30, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 2, 5});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {5, 30, 1080, 3, 640});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 1, 2, 4, 3});
        auto reshape2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {5, 30, 1080, 1920});

        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(input0, reshape0_intput1, false);
        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose0_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape1_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose1_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape2_intput1, false);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape2}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{5, 270, 360, 640});
        auto depth_to_space =
            std::make_shared<ngraph::opset8::DepthToSpace>(input0,
                                                           ngraph::opset8::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                           3);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionFailWithWrongBlockAxisOrder) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 80, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 360, 640, 20, 2, 2});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 4, 2, 3, 1, 5});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 640, 20, 720});
        auto transpose2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 2, 1, 3, 4});
        auto reshape2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1280, 20, 720});
        auto transpose3_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(input0, transpose0_intput1);
        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape0_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose1_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape1_intput1, false);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose2_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose2, reshape2_intput1, false);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose3_intput1);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose3}, ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 80, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 360, 640, 20, 2, 2});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 4, 2, 3, 1, 5});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 640, 20, 720});
        auto transpose2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 2, 1, 3, 4});
        auto reshape2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1280, 20, 720});
        auto transpose3_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(input0, transpose0_intput1);
        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape0_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose1_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape1_intput1, false);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose2_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose2, reshape2_intput1, false);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose3_intput1);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose3}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionDynamicShapeAtInterleaveStart) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{6});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after},
                                               ngraph::ParameterVector{input0, shape_reshape_before});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{6});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after},
                                                   ngraph::ParameterVector{input0, shape_reshape_before});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionDynamicShapeAtInterleaveEnd) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after},
                                               ngraph::ParameterVector{input0, shape_reshape_after});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after},
                                                   ngraph::ParameterVector{input0, shape_reshape_after});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionDynamicShape) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 80, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 360, 640, 20, 2, 2});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 5, 2, 3, 1, 4});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 640, 20, 720});
        auto transpose2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 2, 1, 3, 4});
        auto reshape2_intput1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto transpose3_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(input0, transpose0_intput1);
        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape0_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose1_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape1_intput1, false);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose2_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose2, reshape2_intput1, false);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose3_intput1);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose3},
                                               ngraph::ParameterVector{input0, reshape2_intput1});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 80, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 360, 640, 20, 2, 2});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 5, 2, 3, 1, 4});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 640, 20, 720});
        auto transpose2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 2, 1, 3, 4});
        auto reshape2_intput1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto transpose3_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(input0, transpose0_intput1);
        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape0_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose1_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape1_intput1, false);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose2_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose2, reshape2_intput1, false);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose3_intput1);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose3},
                                                   ngraph::ParameterVector{input0, reshape2_intput1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionSeveralConsumers) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        // additional consumers, not output of the function
        auto result = std::make_shared<ngraph::opset8::Result>(reshape_before);
        auto result_2 = std::make_shared<ngraph::opset8::Result>(permute);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after, result, result_2},
                                               ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset8::Transpose>(reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(permute, shape_reshape_after, false);

        // additional consumers, not output of the function
        auto result = std::make_shared<ngraph::opset8::Result>(reshape_before);
        auto result_2 = std::make_shared<ngraph::opset8::Result>(permute);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after, result, result_2},
                                                   ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, DepthToSpaceFusionMultiConsumerAsStartAndEnd) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 360, 80, 640});
        auto feed_to_multi_input0 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 1, 3});
        auto reshape0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 20, 360, 640});
        auto transpose0_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 2, 5});
        auto reshape1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 20, 720, 2, 640});
        auto transpose1_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {0, 1, 2, 4, 3});
        auto reshape2_intput1 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 20, 720, 1280});
        auto transpose2_input1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto transpose3_input1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto feed_to_multi0 = std::make_shared<ngraph::opset8::Transpose>(input0, feed_to_multi_input0);
        auto reshape0 = std::make_shared<ngraph::opset8::Reshape>(feed_to_multi0, reshape0_intput1, false);
        auto transpose0 = std::make_shared<ngraph::opset8::Transpose>(reshape0, transpose0_intput1);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(transpose0, reshape1_intput1, false);
        auto transpose1 = std::make_shared<ngraph::opset8::Transpose>(reshape1, transpose1_intput1);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(transpose1, reshape2_intput1, false);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose2_input1);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(reshape2, transpose3_input1);

        auto result0 = std::make_shared<ngraph::opset8::Result>(feed_to_multi0);
        auto result1 = std::make_shared<ngraph::opset8::Result>(transpose3);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{result0, transpose2, result1},
                                               ngraph::ParameterVector{input0});

        auto callback = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset8::DepthToSpace>(node) != nullptr;
        };

        ngraph::pass::Manager manager;

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::DepthToSpaceFusion>(callback);

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 360, 80, 640});
        auto feed_to_multi_input0 =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 1, 3});
        auto transpose2_input1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto transpose3_input1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});

        auto feed_to_multi0 = std::make_shared<ngraph::opset8::Transpose>(input0, feed_to_multi_input0);
        auto depth_to_space =
            std::make_shared<ngraph::opset8::DepthToSpace>(feed_to_multi0,
                                                           ngraph::opset8::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                           2);
        auto transpose2 = std::make_shared<ngraph::opset8::Transpose>(depth_to_space, transpose2_input1);
        auto transpose3 = std::make_shared<ngraph::opset8::Transpose>(depth_to_space, transpose3_input1);
        auto result0 = std::make_shared<ngraph::opset8::Result>(feed_to_multi0);
        auto result1 = std::make_shared<ngraph::opset8::Result>(transpose3);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{result0, transpose2, result1},
                                                   ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
