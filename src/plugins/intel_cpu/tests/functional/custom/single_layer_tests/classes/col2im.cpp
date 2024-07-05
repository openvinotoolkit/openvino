// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Col2Im {
std::string Col2ImLayerCPUTest::getTestCaseName(testing::TestParamInfo<Col2ImLayerCPUTestParamsSet> obj) {
        Col2ImLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPrecision;
        ElementType indexPrecision;
        Col2ImSpecificParams col2ImPar;
        std::tie(col2ImPar, netPrecision, indexPrecision, td) = basicParamsSet;

        InputShape inputShape;
        std::vector<int64_t> outputSize;
        std::vector<int64_t> kernelSize;
        ov::Strides strides;
        ov::Strides dilations;
        ov::Shape pads_begin;
        ov::Shape pads_end;
        std::tie(inputShape, outputSize, kernelSize, strides, dilations, pads_begin, pads_end) = col2ImPar;
        std::ostringstream result;

        result << netPrecision << "_IS=";
        result << ov::test::utils::partialShape2str({ inputShape.first }) << "_";
        result << "TS=";
        result << "(";
        for (const auto& targetShape : inputShape.second) {
            result << ov::test::utils::vec2str(targetShape) << "_";
        }
        result << ")_";
        result << "outputSize=" << ov::test::utils::vec2str(outputSize) << "_";
        result << "kernelSize=" << ov::test::utils::vec2str(kernelSize) << "_";
        result << "strides=" << strides << "_";
        result << "dilations=" << dilations << "_";
        result << "padsBegin=" << pads_begin << "_";
        result << "padsEnd=" << pads_end << "_";
        result << "dataPrecision=" << netPrecision << "_";
        result << "indexPrecision=" << indexPrecision;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

void Col2ImLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor data_tensor;
        const auto& dataPrecision = funcInputs[0].get_element_type();
        const auto& dataShape = targetInputStaticShapes.front();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 10;
        in_data.resolution = 1000;
        data_tensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape, in_data);
        inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
    }

void Col2ImLayerCPUTest::SetUp() {
        Col2ImLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        Col2ImSpecificParams Col2ImParams;
        ElementType inputPrecision;
        ElementType indexPrecision;
        std::tie(Col2ImParams, inputPrecision, indexPrecision, targetDevice) = basicParamsSet;

        InputShape dataInputShape;
        std::vector<int64_t> outputSize;
        std::vector<int64_t> kernelSize;
        ov::Strides strides;
        ov::Strides dilations;
        ov::Shape pads_begin;
        ov::Shape pads_end;
        std::tie(dataInputShape, outputSize, kernelSize, strides, dilations, pads_begin, pads_end) = Col2ImParams;

        auto image2DSpatialShape = InputShape{{}, {{2}}};
        init_input_shapes({dataInputShape, image2DSpatialShape, image2DSpatialShape});
        auto dataParameter = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0]);
        auto outputSizeConst = std::make_shared<ov::op::v0::Constant>(indexPrecision, ov::Shape{ 2 }, outputSize);
        auto kernelSizeConst = std::make_shared<ov::op::v0::Constant>(indexPrecision, ov::Shape{ 2 }, kernelSize);

        auto Col2Im = std::make_shared<ov::op::v15::Col2Im>(dataParameter,
                                                            outputSizeConst,
                                                            kernelSizeConst,
                                                            strides,
                                                            dilations,
                                                            pads_begin,
                                                            pads_end);

        ov::ParameterVector params{ dataParameter };
        function = makeNgraphFunction(inputPrecision, params, Col2Im, "Col2Im");

        if (inputPrecision == ov::element::bf16) {
            rel_threshold = 0.02;
        }
}

TEST_P(Col2ImLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Col2Im");
}

const std::vector<Col2ImSpecificParams> col2ImParamsVector = {
    Col2ImSpecificParams {
        InputShape{{}, {{1, 12, 9}}},
        std::vector<int64_t>{4, 4},
        std::vector<int64_t>{2, 2},
        ov::Strides{1, 1},
        ov::Strides{1, 1},
        ov::Shape{0, 0},
        ov::Shape{0, 0}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{3, 12, 81}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{2, 2},
        ov::Shape{2, 2}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{12, 81}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{2, 2},
        ov::Shape{2, 2}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{3, 12, 225}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{1, 1},
        ov::Strides{1, 1},
        ov::Shape{0, 0},
        ov::Shape{0, 0}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{1, 27, 49}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{3, 3},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{1, 1},
        ov::Shape{1, 1}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{1, 18, 104}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 3},
        ov::Strides{2, 1},
        ov::Strides{2, 2},
        ov::Shape{1, 0},
        ov::Shape{0, 1}
    },
    Col2ImSpecificParams {
        InputShape{{-1, -1, -1}, {{1, 12, 120}, {3, 12, 120}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 1},
        ov::Strides{2, 2},
        ov::Shape{1, 0},
        ov::Shape{0, 1}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{-1, 12, 324}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{-1, -1, -1}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{12, -1, -1}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{12, 12, -1}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{12, -1, 324}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{-1, -1}, {{12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    }
};

}  // namespace Col2Im
}  // namespace test
}  // namespace ov
