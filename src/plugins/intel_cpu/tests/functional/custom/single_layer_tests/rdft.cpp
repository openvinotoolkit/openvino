// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/rdft.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::vector<ov::element::Type> precisions{ov::element::f32};

using RDFTTestCPUParams = std::tuple<std::vector<InputShape>,
                                     std::vector<std::vector<int64_t>>,  // axes
                                     std::vector<std::vector<int64_t>>,  // signal sizes
                                     bool,                               // inverse
                                     bool,                               // const axes if true
                                     bool,                               // const signal sizes if true
                                     CPUSpecificParams>;

class RDFTTestCPU : public testing::WithParamInterface<std::tuple<ov::element::Type, RDFTTestCPUParams>>,
                           virtual public test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<ov::element::Type, RDFTTestCPUParams>>& obj) {
        const auto& [precision, params] = obj.param;
        const auto& [shapes, axes, signalSizes, inverse, constAxes, constSignalSizes, cpuParams] = params;
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

        result << ")_isInverse=" << inverse
               << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        inputIdx = 0;
        const auto& [precision, params] = GetParam();
        const auto& [shapes, _axes, _signalSizes, inverse, _constAxes, _constSignalSizes, cpuParams] = params;
        axes = _axes;
        signalSizes = _signalSizes;
        constAxes = _constAxes;
        constSignalSizes = _constSignalSizes;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = makeSelectedTypeStr(selectedType, precision);
        targetDevice = ov::test::utils::DEVICE_CPU;

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

        ParameterVector inputs;
        auto param = std::make_shared<ov::op::v0::Parameter>(precision, *inputShapeIt++);
        inputs.push_back(param);
        std::shared_ptr<Node> axesNode;
        if (constAxes) {
            axesNode = ov::op::v0::Constant::create(element::i64, Shape{axes[0].size()}, axes[0]);
        } else {
            ASSERT_NE(inputShapeIt, inputDynamicShapes.end());
            auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, *inputShapeIt++);
            axesNode = param;
            inputs.push_back(param);
        }

        std::shared_ptr<Node> rdft;
        if (signalSizes.size() > 0) {
            std::shared_ptr<Node> signalSizesNode;
            if (constSignalSizes) {
                signalSizesNode = ov::op::v0::Constant::create(element::i64, Shape{signalSizes[0].size()}, signalSizes[0]);
            } else {
                ASSERT_NE(inputShapeIt, inputDynamicShapes.end());
                auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, *inputShapeIt);
                signalSizesNode = param;
                inputs.push_back(param);
            }
            if (inverse) {
                rdft = std::make_shared<ov::op::v9::IRDFT>(param, axesNode, signalSizesNode);
            } else {
                rdft = std::make_shared<ov::op::v9::RDFT>(param, axesNode, signalSizesNode);
            }
        } else {
            if (inverse) {
                rdft = std::make_shared<ov::op::v9::IRDFT>(param, axesNode);
            } else {
                rdft = std::make_shared<ov::op::v9::RDFT>(param, axesNode);
            }
        }
        function = std::make_shared<Model>(rdft, inputs);

        if (precision == ov::element::f32) {
           abs_threshold = 1e-4;
        }
    }

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        const auto& funcInputs = function->inputs();
        auto funcInput = funcInputs.begin();
        inputs.clear();
        ov::Tensor data_tensor = test::utils::create_and_fill_tensor_normal_distribution(funcInput->get_element_type(),
                                                                                              targetInputStaticShapes[0], 0, 1, 0);

        inputs.insert({funcInput->get_node_shared_ptr(), data_tensor});
        funcInput++;
        if (!constAxes && funcInput != funcInputs.end()) {
            ASSERT_TRUE(inputIdx < axes.size());
            auto tensor = ov::Tensor{funcInput->get_element_type(), Shape{axes[inputIdx].size()}};
            std::memcpy(tensor.data(), axes[inputIdx].data(), axes[inputIdx].size() * sizeof(axes[0][0]));
            inputs.insert({funcInput->get_node_shared_ptr(), tensor});
            funcInput++;
        }
        if (!constSignalSizes && funcInput != funcInputs.end()) {
            ASSERT_TRUE(inputIdx < signalSizes.size());
            auto tensor = ov::Tensor{funcInput->get_element_type(), Shape{signalSizes[inputIdx].size()}};
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

TEST_P(RDFTTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RDFT");
}

namespace {

CPUSpecificParams getCPUSpecificParams() {
    if (ov::with_cpu_x86_avx512_core()) {
        return CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"};
    } else if (ov::with_cpu_x86_avx2()) {
        return CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"};
    } else if (ov::with_cpu_x86_sse42()) {
        return CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"};
    } else {
        return CPUSpecificParams{{}, {}, {"ref"}, "ref"};
    }
    return {};
}

auto cpuParams = getCPUSpecificParams();

std::vector<RDFTTestCPUParams> getParams1D() {
    if (ov::with_cpu_x86_avx512_core()) {
        return {
            {static_shapes_to_test_representation({Shape{14}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{13}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{15}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{30}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{29}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{31}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{46}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{45}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{47}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{126}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{510}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{1022}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{9, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{8, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{10, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{17, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{16, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{18, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{25, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{24, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{26, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{129, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{513, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{1025, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{25, 2}}), {{0}}, {{32}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{24, 2}}), {{0}}, {{16}}, true, true, true, cpuParams},
        };
    } else if (ov::with_cpu_x86_avx2()) {
        return {
            {static_shapes_to_test_representation({Shape{6}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{5}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{7}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{38}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{37}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{39}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{106}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{246}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{245}}), {{0}}, {{118}}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{126}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{510}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{1022}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{5, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{4, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{6, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{9, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{8, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{10, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{17, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{33, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{129, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{257, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{513, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{129, 2}}), {{0}}, {{126}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{257, 2}}), {{0}}, {{254}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{513, 2}}), {{0}}, {{510}}, true, true, true, cpuParams},
        };
    } else {
        return {
            {static_shapes_to_test_representation({Shape{1}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{2}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{12}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{14}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{30}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{62}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{126}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{250}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{254}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{62}}), {{0}}, {{61}}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{126}}), {{0}}, {{40}}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{250}}), {{0}}, {{200}}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{254}}), {{0}}, {{10}}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({Shape{2, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{9, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{10, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{17, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{33, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{65, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{129, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{257, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{33, 2}}), {{0}}, {{50}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{65, 2}}), {{0}}, {{20}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{129, 2}}), {{0}}, {{200}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({Shape{257, 2}}), {{0}}, {{100}}, true, true, true, cpuParams},
        };
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_CPU_1D,
                         RDFTTestCPU,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams1D())),
                         RDFTTestCPU::getTestCaseName);

std::vector<RDFTTestCPUParams> getParams2D() {
    if (ov::with_cpu_x86_avx512_core()) {
        return {
            {static_shapes_to_test_representation({{46, 10}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{45, 10}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{47, 10}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{20, 126}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{20, 510}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{20, 1022}}), {{1}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{48, 46}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 45}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{64, 47}}), {{0, 1}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{72, 126}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 510}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{16, 1022}}), {{0, 1}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{9, 10, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{8, 10, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{10, 20, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{10, 9, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{10, 8, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{20, 10, 2}}), {{1}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{129, 16, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{513, 32, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1025, 72, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 129, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 513, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{72, 1025, 2}}), {{1}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 129, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 513, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{72, 1025, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 129, 2}}), {{0, 1}}, {{16, 200}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 513, 2}}), {{0, 1}}, {{32, 600}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{72, 1025, 2}}), {{0, 1}}, {{72, 100}}, true, true, true, cpuParams},
        };
    } else if (ov::with_cpu_x86_avx2()) {
        return {
            {static_shapes_to_test_representation({{38, 16}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{37, 8}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{39, 24}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 38}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{8, 37}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{24, 39}}), {{1}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 38}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{8, 37}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{24, 39}}), {{0, 1}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{126, 32}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{510, 64}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1022, 64}}), {{0}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{126, 32}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{510, 64}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1022, 64}}), {{0, 1}}, {}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{38, 16, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{37, 8, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{39, 24, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 38, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{8, 37, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{24, 39, 2}}), {{1}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{16, 38, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{8, 37, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{24, 39, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{126, 32, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{510, 64, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1022, 64, 2}}), {{0}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{126, 32, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{510, 64, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1022, 64, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},

            {static_shapes_to_test_representation({{129, 32, 2}}), {{0}}, {{126}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{257, 16, 2}}), {{0}}, {{254}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{513, 64, 2}}), {{0}}, {{510}}, true, true, true, cpuParams},
        };
    } else {
        return {
            {static_shapes_to_test_representation({{1, 1}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 1}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 1}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{13, 13}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{13, 13}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{13, 13}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{29, 29}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{29, 29}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{29, 29}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{30, 32}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 30}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 30}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{62, 64}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{64, 62}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{64, 62}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{254, 128}}), {{0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254}}), {{1}}, {{10}}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254}}), {{0, 1}}, {{128, 100}}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{1, 1, 2}}), {{0}}, {{1}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 1, 2}}), {{1}}, {{1}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 1, 2}}), {{0, 1}}, {{1, 1}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{13, 13, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{13, 13, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{13, 13, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{29, 29, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{29, 29, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{29, 29, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{30, 32, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 30, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{32, 30, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{62, 64, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{64, 62, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{64, 62, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{254, 128, 2}}), {{0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254, 2}}), {{1}}, {{10}}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{128, 254, 2}}), {{0, 1}}, {{128, 100}}, true, true, true, cpuParams},
        };
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_CPU_2D,
                         RDFTTestCPU,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams2D())),
                         RDFTTestCPU::getTestCaseName);

std::vector<RDFTTestCPUParams> getParams4D() {
    std::vector<RDFTTestCPUParams> params;
    if (ov::with_cpu_x86_avx512_core()) {
        params = {
            {static_shapes_to_test_representation({{10, 46, 128, 65}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{10, 46, 128, 65}}), {{0, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65}}), {{1, 0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{10, 46, 128, 65}}), {{1, 2}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65}}), {{-2, -1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65}}), {{3, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65}}), {{0, 1, 2, 3}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65}}), {{0, 1, 2, 3}}, {{10, 10, 33, 50}}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{10, 46, 128, 65, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{10, 46, 128, 65, 2}}), {{0, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65, 2}}), {{1, 0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{10, 46, 128, 65, 2}}), {{1, 2}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65, 2}}), {{-2, -1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65, 2}}), {{3, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{46, 10, 128, 65, 2}}), {{0, 1, 2, 3}}, {}, true, true, true, cpuParams},
            // TODO: FIXME
            //{static_shapes_to_test_representation({{46, 10, 128, 65, 2}}), {{0, 1, 2, 3}, {12, 15, 130, 40}, true, true, true, cpuParams},
        };
    } else if (ov::with_cpu_x86_avx2()) {
        params = {
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{1, 0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{1, 2}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{-2, -1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{3, 1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{0, 1, 2, 3}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126}}), {{0, 1, 2, 3}}, {{8, 10, 11, 12}}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{1, 0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{1, 2}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{-2, -1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{3, 1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{0, 1, 2, 3}}, {}, true, true, true, cpuParams},
            // TODO: FIXME
            //{static_shapes_to_test_representation({{9, 16, 32, 126, 2}}), {{0, 1, 2, 3}}, {{8, 10, 11, 12}}, true, true, true, cpuParams},
        };
    } else {
        params = {
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{1, 0}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{1, 2}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{-2, -1}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{3, 2}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{0, 1, 2, 3}}, {}, false, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30}}), {{0, 1, 2, 3}}, {{1, 2, 3, 13}}, false, true, true, cpuParams},

            {static_shapes_to_test_representation({{1, 2, 13, 30, 2}}), {{1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{2, 2, 13, 30, 2}}), {{1, 0}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30, 2}}), {{1, 2}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30, 2}}), {{-2, -1}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30, 2}}), {{3, 2}}, {}, true, true, true, cpuParams},
            {static_shapes_to_test_representation({{1, 2, 13, 30, 2}}), {{0, 1, 2, 3}}, {}, true, true, true, cpuParams},
            // TODO: FIXME
            //{{1, 2, 13, 30, 2}, {0, 1, 2, 3}, {1, 2, 3, 13}, true, cpuParams},
        };
    }

    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{0}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{1}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{2}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{3}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{0, 1}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{3, 2}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{-2, -1}}, {{36, 64}}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 64}}), {{0, 1, 2, 3}}, {}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 120, 64, 64}}), {{-2, -1}}, {{64, 33}}, false, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 120, 96, 96}}), {{-2, -1}}, {{96, 49}}, false, true, true, cpuParams});

    params.push_back({static_shapes_to_test_representation({{2, 192, 36, 33, 2}}), {{0}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{1}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{2}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{3}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{0, 1}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{3, 2}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{-2, -1}}, {{36, 64}}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 192, 36, 33, 2}}), {{0, 1, 2, 3}}, {}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 120, 64, 33, 2}}), {{-2, -1}}, {{64, 64}}, true, true, true, cpuParams});
    params.push_back({static_shapes_to_test_representation({{1, 120, 96, 49, 2}}), {{-2, -1}}, {{96, 96}}, true, true, true, cpuParams});

    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}}}}, {{0}}, {}, false, true, true, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}, {2, 192, 36, 64}, {3, 192, 36, 64}}},
            InputShape{{-1}, {{1}, {2}, {3}}}}, {{0}, {0, 1}, {1, 2, 3}}, {}, false, false, true, cpuParams});
    params.push_back({{InputShape{{-1, -1, -1, -1}, {{1, 192, 36, 64}, {1, 120, 96, 96}, {2, 120, 96, 96}}},
            InputShape{{-1}, {{1}, {2}, {3}}}}, {{0}, {2, 3}, {0, 2, 3}}, {}, false, false, true, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}, {2, 192, 36, 64}, {3, 192, 36, 64}}}},
            {{2}}, {{12}}, false, true, true, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}, {2, 192, 36, 64}}},
            InputShape{{-1}, {{2}, {3}}}, InputShape{{-1}, {{2}, {3}}}}, {{-2, -1}, {-3, -1, -2}}, {{36, 34}, {192, 64, 30}}, false, false, false, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}, {2, 192, 36, 64}}},
            InputShape{{-1}, {{1}, {2}}}, InputShape{{-1}, {{1}, {2}}}}, {{1}, {-2, -1}}, {{20}, {36, 34}}, false, false, false, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 64}, {{1, 192, 36, 64}, {2, 192, 36, 64}}}, InputShape{{-1}, {{1}, {2}}}, InputShape{{-1}, {{1}, {2}}}},
            {{-1}, {-2, -1}}, {{64}, {36, 34}}, false, false, false, cpuParams});
    params.push_back({{InputShape{{1, 192, -1, -1}, {{1, 192, 36, 64}, {1, 192, 40, 50}}}, InputShape{{-1}, {{2}, {3}}}, InputShape{{-1}, {{2}, {3}}}},
            {{-2, -1}, {2, 1, 3}}, {{36, 34}, {10, 20, 30}}, false, false, false, cpuParams});
    params.push_back({{InputShape{{-1, -1, -1, -1}, {{1, 192, 36, 64}, {1, 120, 96, 96}}}, InputShape{{-1}, {{2}, {2}}}, InputShape{{-1}, {{2}, {2}}}},
            {{-2, -1}, {2, 3}}, {{36, 34}, {96, 49}}, false, false, false, cpuParams});
    params.push_back({{InputShape{{{1, 2}, -1, -1, {1, 100}}, {{1, 192, 36, 64}, {2, 120, 96, 100}}},
            InputShape{{-1}, {{4}, {3}}}, InputShape{{-1}, {{4}, {3}}}},
            {{0, 1, 2, 3}, {-3, -1, -2}}, {{1, 192, 36, 34}, {1, 100, 20}}, false, false, false, cpuParams});

    params.push_back({{InputShape{{-1, 192, 36, 64, 2}, {{2, 192, 36, 64, 2}, {3, 192, 36, 64, 2}}}}, {{0}}, {}, true, true, true, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 33, 2}, {{1, 192, 36, 33, 2}, {2, 192, 36, 33, 2}}}},
            {{-2, -1}}, {{36, 64}}, true, true, true, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 33, 2}, {{1, 192, 36, 33, 2}, {2, 192, 36, 33, 2}}},
            InputShape{{-1}, {{2}, {2}}}, InputShape{{-1}, {{2}, {2}}}}, {{-2, -1}, {-3, -2}}, {{36, 64}, {192, 40}}, true, false, false, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 33, 2}, {{1, 192, 36, 33, 2}, {2, 192, 36, 33, 2}}},
            InputShape{{-1}, {{2}, {3}}}, InputShape{{-1}, {{2}, {3}}}}, {{-2, -1}, {0, 2, 3}}, {{36, 64}, {1, 36, 64}}, true, false, false, cpuParams});
    params.push_back({{InputShape{{-1, 192, 36, 33, 2}, {{1, 192, 36, 33, 2}, {2, 192, 36, 33, 2}}},
            InputShape{{-1}, {{2}, {1}}}, InputShape{{-1}, {{2}, {1}}}},
            {{-2, -1}, {2}}, {{36, 64}, {40}}, true, false, false, cpuParams});
    params.push_back({{InputShape{{-1, 192, -1, -1, 2}, {{1, 192, 36, 33, 2}, {2, 192, 42, 24, 2}}},
            InputShape{{-1}, {{2}, {1}}}, InputShape{{-1}, {{2}, {1}}}},
            {{-2, -1}, {-1}}, {{36, 64}, {30}}, true, false, false, cpuParams});
    params.push_back({{InputShape{{-1, -1, -1, -1, 2}, {{1, 192, 36, 33, 2}, {2, 120, 44, 23, 2}}},
            InputShape{{-1}, {{2}, {1}}}, InputShape{{-1}, {{2}, {1}}}},
            {{-2, -1}, {-2}}, {{36, 64}, {50}}, true, false, false, cpuParams});
    params.push_back({{InputShape{{{1, 2}, -1, -1, {1, 100}, 2}, {{1, 192, 36, 33, 2}, {2, 120, 10, 100, 2}}},
            InputShape{{-1}, {{2}, {1}}}, InputShape{{-1}, {{2}, {1}}}},
            {{-2, -1}, {-1}}, {{36, 64}, {50}}, true, false, false, cpuParams});
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_CPU_4D,
                         RDFTTestCPU,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(getParams4D())),
                         RDFTTestCPU::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
