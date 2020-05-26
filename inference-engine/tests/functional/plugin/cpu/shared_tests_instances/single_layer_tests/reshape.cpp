// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reshape.hpp"
#include "common_test_utils/test_constants.hpp"
#include "../../single_layer_tests/cpu_test_utils.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace {

/********************************************
 * General conformance test case
 ********************************************/

const std::vector<Precision> in_precisions = {
    Precision::FP32,
    Precision::FP16,
    Precision::U8,
    Precision::I8
};

const std::vector<ShapeConfig> shape_param {
    // 0D shape
//  { {},  {} },       // |
//  { {1}, {} },       // | TODO: CNNNetwork has limited support of scalar shape
//  { {1,  1},  {} },  // |
    { {}, {1} },
    { {}, {1, 1} },
    { {}, {1, 1, 1} },

    // 1D shape
    { {10}, {10} },
    { {10}, {10, 1} },
    { {10}, {10, 1, 1} },
    { {10}, {10, 1, 1} },
    { {10}, {1, 10, 1} },
    { {10}, {1, 1, 10} },
    { {10}, {1, 1, -1} },
    { {10}, {2, 5} },
    { {10}, {2, -1} },
    { {10}, {-1, 5} },
    { {10}, {2, 5, -1} },

    // 2D shape
    { {10, 20}, {10, 20} },
    { {10, 20}, {20, 10} },
    { {10, 1},  {10} },

    // 3D shape
    { {10, 20, 5}, {20, 10, 5} },
};

const std::vector<bool> spec_zero = { true, false };
const std::vector<bool> dyn_batch = { false };  // TODO: Dyn batch doesn't works for arbitrary shape.

auto workload_general = ::testing::Combine(
            ::testing::ValuesIn(in_precisions),
            ::testing::ValuesIn(shape_param),
            ::testing::ValuesIn(spec_zero),
            ::testing::ValuesIn(dyn_batch),
            ::testing::Values(CommonTestUtils::DEVICE_CPU));


using ReshapeLayerTestGeneral = ReshapeLayerTest;
TEST_P(ReshapeLayerTestGeneral, CompareWithRefs) {
    Run();
    // Reshape is always memory reinterpret layer, no kernels should be executed
    ASSERT_EQ(0, numOfExecutedNodes(executableNetwork));
}

INSTANTIATE_TEST_CASE_P(General, ReshapeLayerTestGeneral, workload_general, ReshapeLayerTestGeneral::getTestCaseName);

/********************************************
 * Special CPU test case
 ********************************************/

const std::vector<ShapeConfig> shape_param_specific{
        {{3, 32}, {3, 32, 1, 1}},
        {{3, 32}, {3, 32, 1, 1, 1}},

//      {{3, 32}, {3, 32, 1}},             //  |
//      {{3, 32, 6}, {3, 32, 6, 1}},       //  |
//      {{3, 32, 6}, {3, 32, 1, 6}},       //  |
//      {{3, 32, 6}, {3, 32, 2, 3}},       //  |  TODO: The ncw and tnc mismatch.
//      {{3, 32, 6}, {3, 32, 3, 2, 1}},    //  |
//      {{3, 32, 6}, {3, 32, 3, 1, 2}},    //  |
//      {{3, 32, 1}, {3, 32}},             //  |
//      {{3, 32, 4, 6}, {3, 32, 24}},      //  |
//      {{3, 32, 3, 4, 2}, {3, 32, 24}},   //  |

        {{3, 32, 4, 6}, {3, 32, 6, 4}},
        {{3, 32, 4, 6}, {3, 32, 2, 12}},
        {{3, 32, 4, 6}, {3, 32, 6, 4, 1}},
        {{3, 32, 4, 6}, {3, 32, 6, 4, 1}},
        {{3, 32, 1, 1}, {3, 32}},

        {{3, 32, 3, 4, 2}, {3, 32, 6, 4}},
        {{3, 32, 3, 4, 2}, {3, 32, 6, 4, 1}},
        {{3, 32, 3, 4, 2}, {3, 32, 24, 1}},
        {{3, 32, 1, 1, 1}, {3, 32}},
};

auto workload_special = ::testing::Combine(
        ::testing::ValuesIn(in_precisions),
        ::testing::ValuesIn(shape_param_specific),
        ::testing::Values(false),
        ::testing::Values(true, false),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));


class ReshapeLayerTestCPUSpecial : public ReshapeLayerTest {
protected:
    template<typename T>
    void PreSet(const T &&fmtSelector)  {
        auto shapeConf = std::get<1>(this->GetParam());
        inFmt  = fmtSelector(shapeConf.first.size());
        outFmt = fmtSelector(shapeConf.second.size());

        auto ops = function->get_ops();
        auto found = std::find_if(ops.begin(), ops.end(), [] (std::shared_ptr<ngraph::Node> op) {
            return std::string(op->get_type_name()) == "Reshape";
        });
        found->get()->get_rt_info() = setCPUInfo({inFmt}, {outFmt}, {});
    }

    void TearDown() override {
        CheckCPUImpl(executableNetwork, "Reshape", {inFmt}, {outFmt}, "unknown");
        ASSERT_LE(numOfExecutedNodes(executableNetwork), 2);  // TODO: Layout specification doesn't work for in/out nodes, still have reorders
    }

private:
    cpu_memory_format_t inFmt, outFmt;
};

TEST_P(ReshapeLayerTestCPUSpecial, Inplace_TailC) {
    PreSet([] (size_t v) { return tailC_format(v); });
    Run();
}

TEST_P(ReshapeLayerTestCPUSpecial, Inplace_Blocked8C) {
    PreSet([] (size_t v) { return blockedC8_format(v); });
    Run();
}

TEST_P(ReshapeLayerTestCPUSpecial, Inplace_Blocked16C) {
    PreSet([] (size_t v) { return blockedC16_format(v); });
    Run();
}

INSTANTIATE_TEST_CASE_P(Special, ReshapeLayerTestCPUSpecial, workload_special, ReshapeLayerTestCPUSpecial::getTestCaseName);

}  // namespace