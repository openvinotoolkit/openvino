// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/random_uniform.hpp"

namespace {
using ov::test::InputShape;

enum class priorbox_type {
    V0,
    V8,
    Clustered
};
typedef std::tuple<
        InputShape,
        InputShape,
        ov::element::Type,
        std::vector<float>,
        priorbox_type
> PriorBoxLayerGPUTestParamsSet;

class PriorBoxLayerGPUTest : public testing::WithParamInterface<PriorBoxLayerGPUTestParamsSet>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PriorBoxLayerGPUTestParamsSet> obj) {
        InputShape input1Shape;
        InputShape input2Shape;
        ov::element::Type model_type;
        std::vector<float> max_size;
        priorbox_type priorboxType;
        std::tie(input1Shape, input2Shape, model_type, max_size, priorboxType) = obj.param;

        std::ostringstream result;
        switch (priorboxType) {
            case priorbox_type::Clustered:
                result << "PriorBoxClusteredTest_";
                break;
            case priorbox_type::V0:
                result << "PriorBoxV0Test_";
                break;
            case priorbox_type::V8:
            default:
                result << "PriorBoxV8Test_";
        }
        result << std::to_string(obj.index) << "_";
        result << "netPrec=" << model_type << "_";
        result << "I1S=";
        result << ov::test::utils::partialShape2str({input1Shape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : input1Shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")";
        result << "I2S=";
        result << ov::test::utils::partialShape2str({input2Shape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : input2Shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "max_size=" << ov::test::utils::vec2str(max_size) << "_";
        result << ")";
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        auto model_type = ov::element::dynamic;
        InputShape input1Shape;
        InputShape input2Shape;
        std::vector<float> max_size;
        priorbox_type priorboxType;
        std::tie(input1Shape, input2Shape, model_type, max_size, priorboxType) = this->GetParam();


        init_input_shapes({input1Shape, input2Shape});

        inType = ov::element::Type(model_type);
        outType = ov::element::f32;

        auto beginInput = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
        auto endInput = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
        auto strideInput = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});

        ov::ParameterVector functionParams;
        for (auto&& shape : inputDynamicShapes)
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto shapeOfOp1 = std::make_shared<ov::op::v3::ShapeOf>(functionParams[0], ov::element::i32);
        auto shapeOfOp2 = std::make_shared<ov::op::v3::ShapeOf>(functionParams[1], ov::element::i32);

        auto stridedSliceOp1 = std::make_shared<ov::op::v1::StridedSlice>(shapeOfOp1,
                                                                          beginInput,
                                                                          endInput,
                                                                          strideInput,
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{1},
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0});

        auto stridedSliceOp2 = std::make_shared<ov::op::v1::StridedSlice>(shapeOfOp2,
                                                                          beginInput,
                                                                          endInput,
                                                                          strideInput,
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{1},
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0});

        switch (priorboxType) {
            case priorbox_type::Clustered: {
                ov::op::v0::PriorBoxClustered::Attributes attributes_clustered;

                attributes_clustered.widths = {86, 13, 57, 39, 68, 34, 142, 50, 23};
                attributes_clustered.heights = {44, 10, 30, 19, 94, 32, 61, 53, 17};
                attributes_clustered.variances = {0.1, 0.1, 0.2, 0.2};
                attributes_clustered.step = 16;
                attributes_clustered.step_widths = 1;
                attributes_clustered.step_heights = 1;
                attributes_clustered.offset = 0.5;
                attributes_clustered.clip = false;

                auto priorBoxOp = std::make_shared<ov::op::v0::PriorBoxClustered>(stridedSliceOp1, stridedSliceOp2, attributes_clustered);

                ov::ResultVector results{std::make_shared<ov::op::v0::Result>(priorBoxOp)};
                function = std::make_shared <ov::Model>(results, functionParams, "PriorBoxV0Function");
                break;
            }
            case priorbox_type::V0: {
                ov::op::v0::PriorBox::Attributes attributes_v0;

                attributes_v0.min_size = {64};
                attributes_v0.max_size = max_size;
                attributes_v0.aspect_ratio = {2};
                attributes_v0.variance = {0.1, 0.1, 0.2, 0.2};
                attributes_v0.step = 16;
                attributes_v0.offset = 0.5;
                attributes_v0.clip = false;
                attributes_v0.flip = true;
                attributes_v0.scale_all_sizes = true;

                auto priorBoxOp = std::make_shared<ov::op::v0::PriorBox>(stridedSliceOp1, stridedSliceOp2, attributes_v0);

                ov::ResultVector results{std::make_shared<ov::op::v0::Result>(priorBoxOp)};
                function = std::make_shared <ov::Model>(results, functionParams, "PriorBoxV0Function");
                break;
            }
            case priorbox_type::V8:
            default: {
                ov::op::v8::PriorBox::Attributes attributes_v8;

                attributes_v8.min_size = {64};
                attributes_v8.max_size = max_size;
                attributes_v8.aspect_ratio = {2};
                attributes_v8.variance = {0.1, 0.1, 0.2, 0.2};
                attributes_v8.step = 16;
                attributes_v8.offset = 0.5;
                attributes_v8.clip = false;
                attributes_v8.flip = true;
                attributes_v8.scale_all_sizes = true;
                attributes_v8.min_max_aspect_ratios_order = true;

                auto priorBoxOp = std::make_shared<ov::op::v8::PriorBox>(stridedSliceOp1, stridedSliceOp2, attributes_v8);

                ov::ResultVector results{std::make_shared<ov::op::v0::Result>(priorBoxOp)};
                function = std::make_shared <ov::Model>(results, functionParams, "PriorBoxV8Function");
            }
        }
    }
};

TEST_P(PriorBoxLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
};

const std::vector<priorbox_type> mode = {
        priorbox_type::V0,
        priorbox_type::V8,
        // priorboxclustered is not supported yet in prior_box_ref_cl
};

std::vector<ov::test::InputShape> inShapesDynamic = {
        {
            {1, 3, -1, -1},
            {
                { 1, 3, 30, 30 },
                { 1, 3, 20, 20 },
                { 1, 3, 40, 40 }
            }
        },
};
std::vector<ov::test::InputShape> imgShapesDynamic = {
        {
            {1, 3, -1, -1},
            {
                { 1, 3, 224, 224 },
                { 1, 3, 300, 300 },
                { 1, 3, 200, 200 }
            }
        },
};

std::vector<std::vector<float>> max_size = {
        {}, {300}
};
INSTANTIATE_TEST_SUITE_P(smoke_prior_box_full_dynamic,
    PriorBoxLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(imgShapesDynamic),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(max_size),
        ::testing::ValuesIn(mode)),
    PriorBoxLayerGPUTest::getTestCaseName);
} // namespace
