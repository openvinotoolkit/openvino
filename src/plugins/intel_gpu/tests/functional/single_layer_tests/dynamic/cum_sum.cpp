// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/cum_sum.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
    ov::element::Type,      // data type
    InputShape,             // input shape
    std::int64_t,           // axis
    bool,                   // exclusive
    bool                    // reverse
> CumSumLayerGPUParamSet;

class CumSumLayerGPUTest : public testing::WithParamInterface<CumSumLayerGPUParamSet>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CumSumLayerGPUParamSet>& obj) {
        const auto& [model_type, shapes, axis, exclusive, reverse] = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << model_type << "_";
        results << "Axis=" << axis << "_" << (exclusive ? "exclusive" : "") << "_" << (reverse ? "reverse" : "");
        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [model_type, shapes, axis, exclusive, reverse] = this->GetParam();

        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
        }
        auto axisNode = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{axis});
        auto cumSum = std::make_shared<ov::op::v0::CumSum>(params[0], axisNode, exclusive, reverse);

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "CumSumLayerGPUTest");
        };
        function = makeFunction(params, cumSum);
    }
};

TEST_P(CumSumLayerGPUTest, Inference) {
   run();
}

const std::vector<ov::element::Type> model_type = {
    ov::element::f32
};

const std::vector<int64_t> axes = { 0, 1, 2, 3, 4, 5 };
const std::vector<int64_t> negativeAxes = { -1, -2, -3, -4, -5 };

const std::vector<bool> exclusive = { true, false };
const std::vector<bool> reverse = { true, false };

const std::vector<InputShape> inShapes = {
    {{-1},
     {{16}, {18}, {12}}},

    {{-1, -1},
     {{9, 15}, {18, 12}, {12, 12}}},

    {{-1, -1, -1},
     {{16, 10, 12}, {18, 12, 10}, {12, 18, 10}}},

    {{-1, -1, -1, -1},
     {{18, 20, 14, 12}, {19, 20, 14, 12}, {20, 22, 23, 25}}},

    {{-1, -1, -1, -1, -1},
     {{2, 4, 6, 2, 4}, {3, 5, 6, 3, 5}, {1, 4, 2, 6, 8}}},

    {{-1, -1, -1, -1, -1, -1},
     {{2, 4, 6, 2, 4, 2}, {3, 5, 6, 3, 5, 3}, {1, 4, 2, 6, 8, 1}}},
};

const auto testCasesAxis_0 = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(inShapes),
    ::testing::Values(axes[0]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_1 = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 1, inShapes.end())),
    ::testing::Values(axes[1]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_2 = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 2, inShapes.end())),
    ::testing::Values(axes[2]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_3 = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 3, inShapes.end())),
    ::testing::Values(axes[3]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_4 = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 4, inShapes.end())),
    ::testing::Values(axes[4]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_5 = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 5, inShapes.end())),
    ::testing::Values(axes[5]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_negative = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 5, inShapes.end())),
    ::testing::ValuesIn(negativeAxes),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_axis_0_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_0, CumSumLayerGPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_axis_1_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_1, CumSumLayerGPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_axis_2_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_2, CumSumLayerGPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_axis_3_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_3, CumSumLayerGPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_axis_4_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_4, CumSumLayerGPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_axis_5_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_5, CumSumLayerGPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_cum_sum_neg_axes_CompareWithRefs_dynamic, CumSumLayerGPUTest, testCasesAxis_negative, CumSumLayerGPUTest::getTestCaseName);

} // namespace
