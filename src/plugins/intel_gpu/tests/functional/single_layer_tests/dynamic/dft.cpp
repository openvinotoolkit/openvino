// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/idft.hpp"
#include "openvino/op/rdft.hpp"
#include "openvino/op/irdft.hpp"

namespace {
using ov::test::InputShape;

using DFTLayerGPUTestParams = std::tuple<std::vector<InputShape>,
                                    std::vector<std::vector<int64_t>>,  // axes
                                    std::vector<std::vector<int64_t>>,  // signal sizes
                                    bool,                               // inverse
                                    bool,                               // real
                                    bool,                               // const axes if true
                                    bool,                               // const signal sizes if true
                                    std::string>;                       // device name

class DFTLayerGPUTest : public testing::WithParamInterface<std::tuple<ov::element::Type, DFTLayerGPUTestParams>>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<ov::element::Type, DFTLayerGPUTestParams>> obj) {
        ov::element::Type precision;
        DFTLayerGPUTestParams params;
        std::vector<InputShape> shapes;
        std::vector<std::vector<int64_t>> axes;
        std::vector<std::vector<int64_t>> signalSizes;
        bool inverse, real;
        bool constAxes, constSignalSizes;
        std::string targetDevice;

        std::tie(precision, params) = obj.param;
        std::tie(shapes, axes, signalSizes, inverse, real, constAxes, constSignalSizes, targetDevice) = params;

        std::ostringstream result;
        result << "prec=" << precision;
        for (size_t i = 0; i < shapes.size(); i++) {
            result << "_IS" << i << "=" << ov::test::utils::partialShape2str({shapes[i].first});
            result << "_TS" << i << "=(";
            for (size_t j = 0; j < shapes[i].second.size(); j++) {
                result << ov::test::utils::vec2str(shapes[i].second[j]);
                if (j < shapes[i].second.size() - 1)
                    result << "_";
            }
            result << ")";
        }
        result << "_constAxes=" << std::boolalpha << constAxes;
        result << "_axes=(";
        for (size_t i = 0; i < axes.size(); i++) {
            result << ov::test::utils::vec2str(axes[i]);
            if (i < axes.size() - 1)
                result << "_";
        }
        if (signalSizes.size() > 0) {
            result << ")_constSignalSizes=" << std::boolalpha << constSignalSizes;
            result << "_signalSizes=(";
            for (size_t i = 0; i < signalSizes.size(); i++) {
                result << ov::test::utils::vec2str(signalSizes[i]);
                if (i < signalSizes.size() - 1)
                    result << "_";
                }
        }
        result << ")_isInverse=" << inverse;
        result << "_isReal=" << real;
        result << "_device=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type precision;
        DFTLayerGPUTestParams params;
        std::vector<InputShape> shapes;
        bool inverse, real;
        inputIdx = 0;

        std::tie(precision, params) = GetParam();
        std::tie(shapes, axes, signalSizes, inverse, real, constAxes, constSignalSizes, targetDevice) = params;

        if (shapes.size() > 1) {
            ASSERT_EQ(shapes[0].second.size(), shapes[1].second.size());
            if (!constAxes) {
                ASSERT_EQ(shapes[1].second.size(), axes.size());
            }
        }
        if (shapes.size() > 2) {
            ASSERT_EQ(shapes[0].second.size(), shapes[2].second.size());
            if (!constSignalSizes) {
                ASSERT_EQ(shapes[2].second.size(), signalSizes.size());
            }
        }
        if (constAxes) {
            ASSERT_EQ(axes.size(), 1u);
        }
        if (constSignalSizes) {
            ASSERT_LT(signalSizes.size(), 2u);
        }

        init_input_shapes(shapes);

        auto inputShapeIt = inputDynamicShapes.begin();

        ov::ParameterVector inputs;
        auto param = std::make_shared<ov::op::v0::Parameter>(precision, *inputShapeIt++);
        inputs.push_back(param);
        std::shared_ptr<ov::Node> axesNode;
        if (constAxes) {
            axesNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes[0].size()}, axes[0]);
        } else {
            ASSERT_NE(inputShapeIt, inputDynamicShapes.end());
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, *inputShapeIt++);
            axesNode = param;
            inputs.push_back(param);
        }

        std::shared_ptr<ov::Node> dft;
        if (signalSizes.size() > 0) {
            std::shared_ptr<ov::Node> signalSizesNode;
            if (constSignalSizes) {
                signalSizesNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{signalSizes[0].size()}, signalSizes[0]);
            } else {
                ASSERT_NE(inputShapeIt, inputDynamicShapes.end());
                auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, *inputShapeIt);
                signalSizesNode = param;
                inputs.push_back(param);
            }
            if (real) {
                if (inverse) {
                    dft = std::make_shared<ov::op::v9::IRDFT>(param, axesNode, signalSizesNode);
                } else {
                    dft = std::make_shared<ov::op::v9::RDFT>(param, axesNode, signalSizesNode);
                }
            } else {
                if (inverse) {
                    dft = std::make_shared<ov::op::v7::IDFT>(param, axesNode, signalSizesNode);
                } else {
                    dft = std::make_shared<ov::op::v7::DFT>(param, axesNode, signalSizesNode);
                }
            }

        } else {
            if (real) {
                if (inverse) {
                    dft = std::make_shared<ov::op::v9::IRDFT>(param, axesNode);
                } else {
                    dft = std::make_shared<ov::op::v9::RDFT>(param, axesNode);
                }
            } else {
                if (inverse) {
                    dft = std::make_shared<ov::op::v7::IDFT>(param, axesNode);
                } else {
                    dft = std::make_shared<ov::op::v7::DFT>(param, axesNode);
                }
            }
        }
        function = std::make_shared<ov::Model>(dft, inputs);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto& funcInputs = function->inputs();
        auto funcInput = funcInputs.begin();
        inputs.clear();
        ov::Tensor data_tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(funcInput->get_element_type(),
                                                                                              targetInputStaticShapes[0], 0, 1, 0);

        inputs.insert({funcInput->get_node_shared_ptr(), data_tensor});
        funcInput++;
        if (!constAxes && funcInput != funcInputs.end()) {
            ASSERT_TRUE(inputIdx < axes.size());
            auto tensor = ov::Tensor{funcInput->get_element_type(), ov::Shape{axes[inputIdx].size()}};
            std::memcpy(tensor.data(), axes[inputIdx].data(), axes[inputIdx].size() * sizeof(axes[0][0]));
            inputs.insert({funcInput->get_node_shared_ptr(), tensor});
            funcInput++;
        }
        if (!constSignalSizes && funcInput != funcInputs.end()) {
            ASSERT_TRUE(inputIdx < signalSizes.size());
            auto tensor = ov::Tensor{funcInput->get_element_type(), ov::Shape{signalSizes[inputIdx].size()}};
            std::memcpy(tensor.data(), signalSizes[inputIdx].data(), signalSizes[inputIdx].size() * sizeof(signalSizes[0][0]));
            inputs.insert({funcInput->get_node_shared_ptr(), tensor});
        }
        inputIdx++;
    }

    bool constAxes;
    bool constSignalSizes;
    std::vector<std::vector<int64_t>> axes;
    std::vector<std::vector<int64_t>> signalSizes;
    size_t inputIdx = 0;
};

TEST_P(DFTLayerGPUTest, CompareWithRefs) {
    run();
}

std::vector<ov::element::Type> precisions{ov::element::f32, ov::element::f16};

std::vector<DFTLayerGPUTestParams> getParams4D_DFT() {
    std::vector<DFTLayerGPUTestParams> params;

    params.push_back({{InputShape{{-1, 192, 36, 64, 2}, {{2, 192, 36, 64, 2}}}}, {{0}}, {},
            false, false, true, true, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{2}}, {{40}},
            false, false, false, false, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{-2}}, {{40}},
            false, false, false, false, ov::test::utils::DEVICE_GPU});
    return params;
}

std::vector<DFTLayerGPUTestParams> getParams4D_IDFT() {
    std::vector<DFTLayerGPUTestParams> params;

    params.push_back({{InputShape{{-1, 192, 36, 64, 2}, {{2, 192, 36, 64, 2}}}}, {{0}}, {},
            true, false, true, true, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{2}}, {{40}},
            true, false, false, false, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{-2}}, {{40}},
            true, false, false, false, ov::test::utils::DEVICE_GPU});
    return params;
}

std::vector<DFTLayerGPUTestParams> getParams4D_RDFT() {
    std::vector<DFTLayerGPUTestParams> params;

    // RDFT test cases
    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}}}}, {{0}}, {},
            false, true, true, true, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1}, {{1, 192, 36, 64}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{2}}, {{40}},
            false, true, false, false, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1}, {{1, 192, 36, 64}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{-2}}, {{40}},
            false, true, false, false, ov::test::utils::DEVICE_GPU});
    return params;
}

std::vector<DFTLayerGPUTestParams> getParams4D_IRDFT() {
    std::vector<DFTLayerGPUTestParams> params;

    // IRDFT
    params.push_back({{InputShape{{-1, 192, 36, 64, 2}, {{2, 192, 36, 64, 2}}}}, {{0}}, {},
            true, true, true, true, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{2}}, {{40}},
            true, true, false, false, ov::test::utils::DEVICE_GPU});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}}},
            InputShape{{-1}, {{1}}}, InputShape{{-1}, {{1}}}}, {{-2}}, {{40}},
            true, true, false, false, ov::test::utils::DEVICE_GPU});
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_DFT_GPU_4D,
                         DFTLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams4D_DFT())),
                         DFTLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_IDFT_GPU_4D,
                         DFTLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams4D_IDFT())),
                         DFTLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_GPU_4D,
                         DFTLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams4D_RDFT())),
                         DFTLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_GPU_4D,
                         DFTLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams4D_IRDFT())),
                         DFTLayerGPUTest::getTestCaseName);
}  // namespace
