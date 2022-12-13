// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset9.hpp>

using namespace CPUTestUtils;
using namespace ov::test;
using namespace ov;

namespace CPULayerTestsDefinitions {

using RDFTTestCPUParams = std::tuple<
        Shape,
        std::vector<int64_t>,  // axes
        std::vector<int64_t>,  // signal sizes
        bool,                  // inverse
        CPUSpecificParams>;

class RDFTTestCPU : public testing::WithParamInterface<RDFTTestCPUParams>,
                           virtual public test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RDFTTestCPUParams> obj) {
        Shape shape;
        std::vector<int64_t> axes;
        std::vector<int64_t> signalSizes;
        bool inverse;
        CPUSpecificParams cpuParams;

        std::tie(shape, axes, signalSizes, inverse, cpuParams) = obj.param;

        std::ostringstream result;
        result << "shape=" << shape
               << "_axes=" << CommonTestUtils::vec2str(axes)
               << "_signalSizes=" << CommonTestUtils::vec2str(signalSizes)
               << "_isInverse=" << inverse
               << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        Shape shape;
        std::vector<int64_t> axes;
        std::vector<int64_t> signalSizes;
        element::Type_t precision = element::f32;
        bool inverse;
        CPUSpecificParams cpuParams;

        std::tie(shape, axes, signalSizes, inverse, cpuParams) = GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = makeSelectedTypeStr(selectedType, precision);
        targetDevice = CommonTestUtils::DEVICE_CPU;
        targetStaticShapes.push_back(std::vector<Shape>{shape});

        auto param = std::make_shared<opset9::Parameter>(precision, shape);
        auto axesNode = opset9::Constant::create(element::i64, Shape{axes.size()}, axes);
        std::shared_ptr<Node> rdft;
        if (signalSizes.size() > 0) {
            auto signalSizesNode = opset9::Constant::create(element::i64, Shape{signalSizes.size()}, signalSizes);
            if (inverse) {
                rdft = std::make_shared<opset9::IRDFT>(param, axesNode, signalSizesNode);
            } else {
                rdft = std::make_shared<opset9::RDFT>(param, axesNode, signalSizesNode);
            }
        } else {
            if (inverse) {
                rdft = std::make_shared<opset9::IRDFT>(param, axesNode);
            } else {
                rdft = std::make_shared<opset9::RDFT>(param, axesNode);
            }
        }
        function = std::make_shared<Model>(rdft, ParameterVector{param});
    }

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        const auto& funcInputs = function->inputs();
        inputs.clear();

        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            runtime::Tensor tensor = test::utils::create_and_fill_tensor_normal_distribution(funcInput.get_element_type(), targetInputStaticShapes[0], 0, 1, 0);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(RDFTTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RDFT");
}

namespace {

CPUSpecificParams getCPUSpecificParams() {
    if (InferenceEngine::with_cpu_x86_avx512_core()) {
        return CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"};
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        return CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"};
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        return CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"};
    } else {
        return CPUSpecificParams{{}, {}, {"ref"}, "ref"};
    }
    return {};
}

auto cpuParams = getCPUSpecificParams();

std::vector<RDFTTestCPUParams> getParams1D() {
    if (InferenceEngine::with_cpu_x86_avx512_core()) {
        return {
            {{14}, {0}, {}, false, cpuParams},
            {{13}, {0}, {}, false, cpuParams},
            {{15}, {0}, {}, false, cpuParams},

            {{30}, {0}, {}, false, cpuParams},
            {{29}, {0}, {}, false, cpuParams},
            {{31}, {0}, {}, false, cpuParams},

            {{46}, {0}, {}, false, cpuParams},
            {{45}, {0}, {}, false, cpuParams},
            {{47}, {0}, {}, false, cpuParams},

            {{126}, {0}, {}, false, cpuParams},
            {{510}, {0}, {}, false, cpuParams},
            {{1022}, {0}, {}, false, cpuParams},

            {{9, 2}, {0}, {}, true, cpuParams},
            {{8, 2}, {0}, {}, true, cpuParams},
            {{10, 2}, {0}, {}, true, cpuParams},

            {{17, 2}, {0}, {}, true, cpuParams},
            {{16, 2}, {0}, {}, true, cpuParams},
            {{18, 2}, {0}, {}, true, cpuParams},

            {{25, 2}, {0}, {}, true, cpuParams},
            {{24, 2}, {0}, {}, true, cpuParams},
            {{26, 2}, {0}, {}, true, cpuParams},

            {{129, 2}, {0}, {}, true, cpuParams},
            {{513, 2}, {0}, {}, true, cpuParams},
            {{1025, 2}, {0}, {}, true, cpuParams},

            {{25, 2}, {0}, {32}, true, cpuParams},
            {{24, 2}, {0}, {16}, true, cpuParams},
        };
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        return {
            {{6}, {0}, {}, false, cpuParams},
            {{5}, {0}, {}, false, cpuParams},
            {{7}, {0}, {}, false, cpuParams},

            {{38}, {0}, {}, false, cpuParams},
            {{37}, {0}, {}, false, cpuParams},
            {{39}, {0}, {}, false, cpuParams},

            {{106}, {0}, {}, false, cpuParams},
            {{246}, {0}, {}, false, cpuParams},
            {{245}, {0}, {118}, false, cpuParams},

            {{126}, {0}, {}, false, cpuParams},
            {{510}, {0}, {}, false, cpuParams},
            {{1022}, {0}, {}, false, cpuParams},

            {{5, 2}, {0}, {}, true, cpuParams},
            {{4, 2}, {0}, {}, true, cpuParams},
            {{6, 2}, {0}, {}, true, cpuParams},

            {{9, 2}, {0}, {}, true, cpuParams},
            {{8, 2}, {0}, {}, true, cpuParams},
            {{10, 2}, {0}, {}, true, cpuParams},

            {{17, 2}, {0}, {}, true, cpuParams},
            {{33, 2}, {0}, {}, true, cpuParams},
            {{129, 2}, {0}, {}, true, cpuParams},
            {{257, 2}, {0}, {}, true, cpuParams},
            {{513, 2}, {0}, {}, true, cpuParams},

            {{129, 2}, {0}, {126}, true, cpuParams},
            {{257, 2}, {0}, {254}, true, cpuParams},
            {{513, 2}, {0}, {510}, true, cpuParams},
        };
    } else {
        return {
            {{1}, {0}, {}, false, cpuParams},
            {{2}, {0}, {}, false, cpuParams},
            {{12}, {0}, {}, false, cpuParams},
            {{14}, {0}, {}, false, cpuParams},
            {{30}, {0}, {}, false, cpuParams},
            {{62}, {0}, {}, false, cpuParams},
            {{126}, {0}, {}, false, cpuParams},
            {{250}, {0}, {}, false, cpuParams},
            {{254}, {0}, {}, false, cpuParams},
            {{62}, {0}, {61}, false, cpuParams},
            {{126}, {0}, {40}, false, cpuParams},
            {{250}, {0}, {200}, false, cpuParams},
            {{254}, {0}, {10}, false, cpuParams},

            {{2, 2}, {0}, {}, true, cpuParams},
            {{9, 2}, {0}, {}, true, cpuParams},
            {{10, 2}, {0}, {}, true, cpuParams},
            {{17, 2}, {0}, {}, true, cpuParams},
            {{33, 2}, {0}, {}, true, cpuParams},
            {{65, 2}, {0}, {}, true, cpuParams},
            {{129, 2}, {0}, {}, true, cpuParams},
            {{257, 2}, {0}, {}, true, cpuParams},
            {{33, 2}, {0}, {50}, true, cpuParams},
            {{65, 2}, {0}, {20}, true, cpuParams},
            {{129, 2}, {0}, {200}, true, cpuParams},
            {{257, 2}, {0}, {100}, true, cpuParams},
        };
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_CPU_1D, RDFTTestCPU, ::testing::ValuesIn(getParams1D()), RDFTTestCPU::getTestCaseName);

std::vector<RDFTTestCPUParams> getParams2D() {
    if (InferenceEngine::with_cpu_x86_avx512_core()) {
        return {
            {{46, 10}, {0}, {}, false, cpuParams},
            {{45, 10}, {0}, {}, false, cpuParams},
            {{47, 10}, {0}, {}, false, cpuParams},

            {{20, 126}, {1}, {}, false, cpuParams},
            {{20, 510}, {1}, {}, false, cpuParams},
            {{20, 1022}, {1}, {}, false, cpuParams},

            {{48, 46}, {0, 1}, {}, false, cpuParams},
            {{32, 45}, {0, 1}, {}, false, cpuParams},
            {{64, 47}, {0, 1}, {}, false, cpuParams},

            {{72, 126}, {0, 1}, {}, false, cpuParams},
            {{32, 510}, {0, 1}, {}, false, cpuParams},
            {{16, 1022}, {0, 1}, {}, false, cpuParams},

            {{9, 10, 2}, {0}, {}, true, cpuParams},
            {{8, 10, 2}, {0}, {}, true, cpuParams},
            {{10, 20, 2}, {0}, {}, true, cpuParams},

            {{10, 9, 2}, {1}, {}, true, cpuParams},
            {{10, 8, 2}, {1}, {}, true, cpuParams},
            {{20, 10, 2}, {1}, {}, true, cpuParams},

            {{129, 16, 2}, {0}, {}, true, cpuParams},
            {{513, 32, 2}, {0}, {}, true, cpuParams},
            {{1025, 72, 2}, {0}, {}, true, cpuParams},

            {{16, 129, 2}, {1}, {}, true, cpuParams},
            {{32, 513, 2}, {1}, {}, true, cpuParams},
            {{72, 1025, 2}, {1}, {}, true, cpuParams},

            {{16, 129, 2}, {0, 1}, {}, true, cpuParams},
            {{32, 513, 2}, {0, 1}, {}, true, cpuParams},
            {{72, 1025, 2}, {0, 1}, {}, true, cpuParams},

            {{16, 129, 2}, {0, 1}, {16, 200}, true, cpuParams},
            {{32, 513, 2}, {0, 1}, {32, 600}, true, cpuParams},
            {{72, 1025, 2}, {0, 1}, {72, 100}, true, cpuParams},
        };
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        return {
            {{38, 16}, {0}, {}, false, cpuParams},
            {{37, 8}, {0}, {}, false, cpuParams},
            {{39, 24}, {0}, {}, false, cpuParams},

            {{16, 38}, {1}, {}, false, cpuParams},
            {{8, 37}, {1}, {}, false, cpuParams},
            {{24, 39}, {1}, {}, false, cpuParams},

            {{16, 38}, {0, 1}, {}, false, cpuParams},
            {{8, 37}, {0, 1}, {}, false, cpuParams},
            {{24, 39}, {0, 1}, {}, false, cpuParams},

            {{126, 32}, {0}, {}, false, cpuParams},
            {{510, 64}, {0}, {}, false, cpuParams},
            {{1022, 64}, {0}, {}, false, cpuParams},

            {{126, 32}, {0, 1}, {}, false, cpuParams},
            {{510, 64}, {0, 1}, {}, false, cpuParams},
            {{1022, 64}, {0, 1}, {}, false, cpuParams},

            {{38, 16, 2}, {0}, {}, true, cpuParams},
            {{37, 8, 2}, {0}, {}, true, cpuParams},
            {{39, 24, 2}, {0}, {}, true, cpuParams},

            {{16, 38, 2}, {1}, {}, true, cpuParams},
            {{8, 37, 2}, {1}, {}, true, cpuParams},
            {{24, 39, 2}, {1}, {}, true, cpuParams},

            {{16, 38, 2}, {0, 1}, {}, true, cpuParams},
            {{8, 37, 2}, {0, 1}, {}, true, cpuParams},
            {{24, 39, 2}, {0, 1}, {}, true, cpuParams},

            {{126, 32, 2}, {0}, {}, true, cpuParams},
            {{510, 64, 2}, {0}, {}, true, cpuParams},
            {{1022, 64, 2}, {0}, {}, true, cpuParams},

            {{126, 32, 2}, {0, 1}, {}, true, cpuParams},
            {{510, 64, 2}, {0, 1}, {}, true, cpuParams},
            {{1022, 64, 2}, {0, 1}, {}, true, cpuParams},

            {{129, 32, 2}, {0}, {126}, true, cpuParams},
            {{257, 16, 2}, {0}, {254}, true, cpuParams},
            {{513, 64, 2}, {0}, {510}, true, cpuParams},
        };
    } else {
        return {
            {{1, 1}, {0}, {}, false, cpuParams},
            {{1, 1}, {1}, {}, false, cpuParams},
            {{1, 1}, {0, 1}, {}, false, cpuParams},
            {{2, 2}, {0}, {}, false, cpuParams},
            {{2, 2}, {1}, {}, false, cpuParams},
            {{2, 2}, {0, 1}, {}, false, cpuParams},
            {{13, 13}, {0}, {}, false, cpuParams},
            {{13, 13}, {1}, {}, false, cpuParams},
            {{13, 13}, {0, 1}, {}, false, cpuParams},
            {{29, 29}, {0}, {}, false, cpuParams},
            {{29, 29}, {1}, {}, false, cpuParams},
            {{29, 29}, {0, 1}, {}, false, cpuParams},
            {{30, 32}, {0}, {}, false, cpuParams},
            {{32, 30}, {1}, {}, false, cpuParams},
            {{32, 30}, {0, 1}, {}, false, cpuParams},
            {{62, 64}, {0}, {}, false, cpuParams},
            {{64, 62}, {1}, {}, false, cpuParams},
            {{64, 62}, {0, 1}, {}, false, cpuParams},
            {{254, 128}, {0}, {}, false, cpuParams},
            {{128, 254}, {1}, {}, false, cpuParams},
            {{128, 254}, {0, 1}, {}, false, cpuParams},
            {{128, 254}, {1}, {10}, false, cpuParams},
            {{128, 254}, {0, 1}, {128, 100}, false, cpuParams},

            {{1, 1, 2}, {0}, {1}, true, cpuParams},
            {{1, 1, 2}, {1}, {1}, true, cpuParams},
            {{1, 1, 2}, {0, 1}, {1, 1}, true, cpuParams},
            {{2, 2, 2}, {0}, {}, true, cpuParams},
            {{2, 2, 2}, {1}, {}, true, cpuParams},
            {{2, 2, 2}, {0, 1}, {}, true, cpuParams},
            {{13, 13, 2}, {0}, {}, true, cpuParams},
            {{13, 13, 2}, {1}, {}, true, cpuParams},
            {{13, 13, 2}, {0, 1}, {}, true, cpuParams},
            {{29, 29, 2}, {0}, {}, true, cpuParams},
            {{29, 29, 2}, {1}, {}, true, cpuParams},
            {{29, 29, 2}, {0, 1}, {}, true, cpuParams},
            {{30, 32, 2}, {0}, {}, true, cpuParams},
            {{32, 30, 2}, {1}, {}, true, cpuParams},
            {{32, 30, 2}, {0, 1}, {}, true, cpuParams},
            {{62, 64, 2}, {0}, {}, true, cpuParams},
            {{64, 62, 2}, {1}, {}, true, cpuParams},
            {{64, 62, 2}, {0, 1}, {}, true, cpuParams},
            {{254, 128, 2}, {0}, {}, true, cpuParams},
            {{128, 254, 2}, {1}, {}, true, cpuParams},
            {{128, 254, 2}, {0, 1}, {}, true, cpuParams},
            {{128, 254, 2}, {1}, {10}, true, cpuParams},
            {{128, 254, 2}, {0, 1}, {128, 100}, true, cpuParams},
        };
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_CPU_2D, RDFTTestCPU, ::testing::ValuesIn(getParams2D()), RDFTTestCPU::getTestCaseName);


std::vector<RDFTTestCPUParams> getParams4D() {
    std::vector<RDFTTestCPUParams> params;
    if (InferenceEngine::with_cpu_x86_avx512_core()) {
        params = {
            {{10, 46, 128, 65}, {1}, {}, false, cpuParams},
            {{10, 46, 128, 65}, {0, 1}, {}, false, cpuParams},
            {{46, 10, 128, 65}, {1, 0}, {}, false, cpuParams},
            {{10, 46, 128, 65}, {1, 2}, {}, false, cpuParams},
            {{46, 10, 128, 65}, {-2, -1}, {}, false, cpuParams},
            {{46, 10, 128, 65}, {3, 1}, {}, false, cpuParams},
            {{46, 10, 128, 65}, {0, 1, 2, 3}, {}, false, cpuParams},
            {{46, 10, 128, 65}, {0, 1, 2, 3}, {10, 10, 33, 50}, false, cpuParams},

            {{10, 46, 128, 65, 2}, {1}, {}, true, cpuParams},
            {{10, 46, 128, 65, 2}, {0, 1}, {}, true, cpuParams},
            {{46, 10, 128, 65, 2}, {1, 0}, {}, true, cpuParams},
            {{10, 46, 128, 65, 2}, {1, 2}, {}, true, cpuParams},
            {{46, 10, 128, 65, 2}, {-2, -1}, {}, true, cpuParams},
            {{46, 10, 128, 65, 2}, {3, 1}, {}, true, cpuParams},
            {{46, 10, 128, 65, 2}, {0, 1, 2, 3}, {}, true, cpuParams},
            // TODO: FIXME
            //{{46, 10, 128, 65, 2}, {0, 1, 2, 3}, {12, 15, 130, 40}, true, cpuParams},
        };
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        params = {
            {{9, 16, 32, 126}, {1}, {}, false, cpuParams},
            {{9, 16, 32, 126}, {1, 0}, {}, false, cpuParams},
            {{9, 16, 32, 126}, {1, 2}, {}, false, cpuParams},
            {{9, 16, 32, 126}, {-2, -1}, {}, false, cpuParams},
            {{9, 16, 32, 126}, {3, 1}, {}, false, cpuParams},
            {{9, 16, 32, 126}, {0, 1, 2, 3}, {}, false, cpuParams},
            {{9, 16, 32, 126}, {0, 1, 2, 3}, {8, 10, 11, 12}, false, cpuParams},

            {{9, 16, 32, 126, 2}, {1}, {}, true, cpuParams},
            {{9, 16, 32, 126, 2}, {1, 0}, {}, true, cpuParams},
            {{9, 16, 32, 126, 2}, {1, 2}, {}, true, cpuParams},
            {{9, 16, 32, 126, 2}, {-2, -1}, {}, true, cpuParams},
            {{9, 16, 32, 126, 2}, {3, 1}, {}, true, cpuParams},
            {{9, 16, 32, 126, 2}, {0, 1, 2, 3}, {}, true, cpuParams},
            // TODO: FIXME
            //{{9, 16, 32, 126, 2}, {0, 1, 2, 3}, {8, 10, 11, 12}, true, cpuParams},
        };
    } else {
        params = {
            {{1, 2, 13, 30}, {1}, {}, false, cpuParams},
            {{1, 2, 13, 30}, {1, 0}, {}, false, cpuParams},
            {{1, 2, 13, 30}, {1, 2}, {}, false, cpuParams},
            {{1, 2, 13, 30}, {-2, -1}, {}, false, cpuParams},
            {{1, 2, 13, 30}, {3, 2}, {}, false, cpuParams},
            {{1, 2, 13, 30}, {0, 1, 2, 3}, {}, false, cpuParams},
            {{1, 2, 13, 30}, {0, 1, 2, 3}, {1, 2, 3, 13}, false, cpuParams},

            {{1, 2, 13, 30, 2}, {1}, {}, true, cpuParams},
            {{2, 2, 13, 30, 2}, {1, 0}, {}, true, cpuParams},
            {{1, 2, 13, 30, 2}, {1, 2}, {}, true, cpuParams},
            {{1, 2, 13, 30, 2}, {-2, -1}, {}, true, cpuParams},
            {{1, 2, 13, 30, 2}, {3, 2}, {}, true, cpuParams},
            {{1, 2, 13, 30, 2}, {0, 1, 2, 3}, {}, true, cpuParams},
            // TODO: FIXME
            //{{1, 2, 13, 30, 2}, {0, 1, 2, 3}, {1, 2, 3, 13}, true, cpuParams},
        };
    }
    params.push_back({{1, 192, 36, 64}, {0}, {}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {1}, {}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {2}, {}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {3}, {}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {0, 1}, {}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {3, 2}, {}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {-2, -1}, {36, 64}, false, cpuParams});
    params.push_back({{1, 192, 36, 64}, {0, 1, 2, 3}, {}, false, cpuParams});
    params.push_back({{2, 192, 36, 33, 2}, {0}, {}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {1}, {}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {2}, {}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {3}, {}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {0, 1}, {}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {3, 2}, {}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {-2, -1}, {36, 64}, true, cpuParams});
    params.push_back({{1, 192, 36, 33, 2}, {0, 1, 2, 3}, {}, true, cpuParams});

    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_CPU_4D, RDFTTestCPU, ::testing::ValuesIn(getParams4D()), RDFTTestCPU::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
