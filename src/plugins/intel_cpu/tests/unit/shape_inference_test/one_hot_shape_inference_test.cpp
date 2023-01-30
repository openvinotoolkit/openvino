// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "one_hot_shape_inference.hpp"

#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "utils/shape_inference/shape_inference.hpp"
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

TEST(StaticShapeInferenceTest, OneHotTestConstantInput) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto on_value = op::v0::Constant::create(element::u32, Shape{}, {5});
    auto off_value = op::v0::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(ont_hot.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMap) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto on_param = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto off_param = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_param, off_param, axis);

    int64_t depth_value[] = {2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i64, Shape{}, depth_value);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, on_value);
    constant_data[3] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, off_value);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(ont_hot.get(), static_input_shapes, static_output_shapes, constant_data);
    EXPECT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMapDefaultCtor) {
    auto ont_hot = std::make_shared<op::v1::OneHot>();
    ont_hot->set_axis(-1);

    int64_t depth_value[] = {2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i64, Shape{}, depth_value);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, on_value);
    constant_data[3] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, off_value);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};

    shape_infer(ont_hot.get(), static_input_shapes, static_output_shapes, constant_data);

    EXPECT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMapNegativeDepth) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto on_param = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto off_param = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_param, off_param, axis);

    int64_t depth_value[] = {-2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i64, Shape{}, depth_value);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, on_value);
    constant_data[3] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, off_value);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(shape_inference(ont_hot.get(), static_input_shapes, static_output_shapes, constant_data),
                    ov::NodeValidationFailure,
                    HasSubstr("can't be negative"));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMapDefaultCtorPartialShape) {
    auto ont_hot = std::make_shared<op::v1::OneHot>();
    ont_hot->set_axis(-1);

    int64_t depth_value[] = {2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i64, Shape{}, depth_value);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, on_value);
    constant_data[3] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, off_value);

    std::vector<PartialShape> input_shapes = {PartialShape{3}, PartialShape{}, PartialShape{}, PartialShape{}},
                              output_shapes = {PartialShape{}};

    shape_infer(ont_hot.get(), input_shapes, output_shapes, constant_data);

    EXPECT_EQ(output_shapes[0], (PartialShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMapDefaultCtorNegativeDepthPartialShape) {
    auto ont_hot = std::make_shared<op::v1::OneHot>();
    ont_hot->set_axis(-1);

    int64_t depth_value[] = {-2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i64, Shape{}, depth_value);
    constant_data[2] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, on_value);
    constant_data[3] =
        std::make_shared<ngraph::runtime::HostTensor>(element::Type_t::i32, Shape{}, off_value);

    std::vector<PartialShape> input_shapes = {PartialShape{3}, PartialShape{}, PartialShape{}, PartialShape{}},
                              output_shapes = {PartialShape{}};

    OV_EXPECT_THROW(shape_infer(ont_hot.get(), input_shapes, output_shapes, constant_data),
                    ov::NodeValidationFailure,
                    HasSubstr("can't be negative"));
}
