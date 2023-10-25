// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "any_copy.hpp"
#include "backend/gna_limitations.hpp"
#include "common/gna_target.hpp"
#include "gna_data_types.hpp"
#include "gna_plugin.hpp"
#include "memory/gna_memory.hpp"
#include "ov_models/builders.hpp"

using namespace InferenceEngine;
using namespace ov::intel_gna::target;
using namespace ov::intel_gna::limitations;
namespace testing {

using MemAlignmentTestParams =
    std::tuple<ExecutionMode,                    // execution mode
               HWGeneration,                     // compile target
               std::pair<ngraph::Shape, size_t>  // input shape vs expected memory size of the input region in bytes.
                                                 // For this specific model and when the value of input_shape_H = 1,
                                                 // the memory input region size can be calculated using below formula:
                                                 // mem_input_region_size = ALIGN8(input_shape_W)*inputPrecInBytes.
                                                 // Refer to GNAGraphCompiler::AffinePrimitive for more details.
               >;

const std::vector<std::pair<ngraph::Shape, size_t>> param_16B_alignment_prec_fp32{{{1, 2}, 32},
                                                                                  {{1, 8}, 32},
                                                                                  {{1, 9}, 64}};

const std::vector<std::pair<ngraph::Shape, size_t>> param_64B_alignment_prec_int16{{{1, 2}, 64},
                                                                                   {{1, 32}, 64},
                                                                                   {{1, 33}, 128}};

const std::vector<std::pair<ngraph::Shape, size_t>> param_16B_alignment_prec_int16{{{1, 2}, 16},
                                                                                   {{1, 8}, 16},
                                                                                   {{1, 9}, 32},
                                                                                   {{1, 33}, 80}};

class GNAPluginForMemoryAlignmentTest : public GNAPlugin {
public:
    GNAPluginForMemoryAlignmentTest(const std::map<std::string, std::string>& configMap) : GNAPlugin(configMap) {
        if (gnadevice) {
            gnamem.reset(new gna_memory_float(memory::GNAFloatAllocator{},
                                              Limitations::get_instance()->get_memory_alignment(),
                                              Limitations::kMemoryPageSize));
            m_graph_compiler->setGNAMemoryPtr(gnamem);
            gnadevice.reset();
        }
    }

    const size_t get_memory_REGION_INPUTS_size() const {
        return this->gnamem->getQueue(ov::intel_gna::memory::REGION_INPUTS)->calcSize();
    }
};

class GNAPluginLoadNetworkTests : public ::testing::TestWithParam<MemAlignmentTestParams> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<MemAlignmentTestParams>& obj) {
        ExecutionMode exe_mode;
        HWGeneration hw_gen;
        std::pair<ngraph::Shape, size_t> inp_shape_vs_mem;
        tie(exe_mode, hw_gen, inp_shape_vs_mem) = obj.param;

        std::ostringstream result;
        result << "inp=" << inp_shape_vs_mem.first.to_string() << "_";
        result << "mem_region_size=" << inp_shape_vs_mem.second;
        return result.str();
    }

protected:
    void Run() {
        ExecutionMode exe_mode;
        HWGeneration hw_gen;
        std::pair<ngraph::Shape, size_t> inp_shape_vs_mem;
        tie(exe_mode, hw_gen, inp_shape_vs_mem) = this->GetParam();
        ngraph::Shape inp_shape = inp_shape_vs_mem.first;
        size_t mem_region_size = inp_shape_vs_mem.second;

        const ov::AnyMap gna_config = {ov::intel_gna::execution_mode(exe_mode), ov::intel_gna::compile_target(hw_gen)};

        auto plugin = GNAPluginForMemoryAlignmentTest(any_copy(gna_config));
        auto function = getMulFunction(inp_shape);
        CNNNetwork cnnNetwork(function);
        plugin.LoadNetwork(cnnNetwork);
        EXPECT_EQ(plugin.get_memory_REGION_INPUTS_size(), mem_region_size);
    }

    void SetUp() override {
        test_params = GetParam();
    }

private:
    std::shared_ptr<ov::Model> getMulFunction(const ngraph::Shape input_shape) {
        const ngraph::element::Type net_precision = ngraph::element::f32;

        auto input = std::make_shared<ngraph::opset8::Parameter>(net_precision, input_shape);
        auto multiplier = std::make_shared<ngraph::opset8::Constant>(net_precision, input_shape);
        auto matmul = std::make_shared<ngraph::opset8::MatMul>(input, multiplier, false, true);
        auto result = std::make_shared<ngraph::opset8::Result>(matmul);
        auto function = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "MatMul");
        return function;
    }

    MemAlignmentTestParams test_params;
};

TEST_P(GNAPluginLoadNetworkTests, CompareInpShapeVsReservedMemRegion) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(MemoryAlignment_FP32,
                         GNAPluginLoadNetworkTests,
                         ::testing::Combine(::testing::Values(ExecutionMode::SW_FP32),
                                            ::testing::Values(HWGeneration::UNDEFINED),
                                            ::testing::ValuesIn(param_16B_alignment_prec_fp32)),
                         GNAPluginLoadNetworkTests::GetTestCaseName);

INSTANTIATE_TEST_SUITE_P(MemoryAlignment_GNA_3_0,
                         GNAPluginLoadNetworkTests,
                         ::testing::Combine(::testing::Values(ExecutionMode::SW_EXACT),
                                            ::testing::Values(HWGeneration::GNA_3_0),
                                            ::testing::ValuesIn(param_64B_alignment_prec_int16)),
                         GNAPluginLoadNetworkTests::GetTestCaseName);

INSTANTIATE_TEST_SUITE_P(MemoryAlignment_GNA_3_5,
                         GNAPluginLoadNetworkTests,
                         ::testing::Combine(::testing::Values(ExecutionMode::SW_EXACT),
                                            ::testing::Values(HWGeneration::GNA_3_5),
                                            ::testing::ValuesIn(param_64B_alignment_prec_int16)),
                         GNAPluginLoadNetworkTests::GetTestCaseName);

INSTANTIATE_TEST_SUITE_P(MemoryAlignment_GNA_3_6,
                         GNAPluginLoadNetworkTests,
                         ::testing::Combine(::testing::Values(ExecutionMode::SW_EXACT),
                                            ::testing::Values(HWGeneration::GNA_3_6),
                                            ::testing::ValuesIn(param_16B_alignment_prec_int16)),
                         GNAPluginLoadNetworkTests::GetTestCaseName);

INSTANTIATE_TEST_SUITE_P(MemoryAlignment_GNA_4_0,
                         GNAPluginLoadNetworkTests,
                         ::testing::Combine(::testing::Values(ExecutionMode::SW_EXACT),
                                            ::testing::Values(HWGeneration::GNA_4_0),
                                            ::testing::ValuesIn(param_16B_alignment_prec_int16)),
                         GNAPluginLoadNetworkTests::GetTestCaseName);

class MemoryAlignmentTest : public ::testing::Test {};

TEST(MemoryAlignmentTest, getMemoryAlignmentBytes_Expect64ByteAlignmentWhenTargetIsGNA3_5) {
    Limitations::init(DeviceVersion::GNA3_5);
    EXPECT_EQ(Limitations::get_instance()->get_memory_alignment(), 64);
}

TEST(MemoryAlignmentTest, getMemoryAlignmentBytes_Expect16ByteAlignmentWhenTargetIsGNA3_6) {
    Limitations::init(DeviceVersion::GNA3_6);
    EXPECT_EQ(Limitations::get_instance()->get_memory_alignment(), 16);
}

}  // namespace testing
