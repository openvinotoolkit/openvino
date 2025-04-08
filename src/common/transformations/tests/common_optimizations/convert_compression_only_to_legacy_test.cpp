// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST(TransformationTests, ConvertCompressionOnlyToLegacy) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        ov::mark_as_decompression(convert_ins1);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertCompressedOnlyToLegacy>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertCompressionOnlyToLegacyNoConvertion) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertCompressedOnlyToLegacy>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
