// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "any_copy.hpp"
#include "backend/gna_limitations.hpp"
#include "common/gna_target.hpp"
#include "gna_data_types.hpp"
#include "gna_plugin.hpp"
#include "memory/gna_memory.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace ov::intel_gna::target;
using namespace ov::intel_gna::limitations;
namespace testing {

struct MemAlignmentTestParams {
    ExecutionMode execution_mode;
    HWGeneration compile_target;
    ngraph::Shape input_shape;
    size_t mem_input_region_size;  // Expected memory size of the input region in bytes. For this specific model and
                                   // when the value of input_shape_H is 1, the memory input region size can be
                                   // calculated using below formula:
                                   // mem_input_region_size = ALIGN8(input_shape_W)*inputPrecInBytes.
                                   // Refer to GNAGraphCompiler::AffinePrimitive for more details.
};

std::vector<MemAlignmentTestParams> all_tests_params{
    {ExecutionMode::SW_FP32, HWGeneration::UNDEFINED, {1, 2}, 32},
    {ExecutionMode::SW_FP32, HWGeneration::UNDEFINED, {1, 8}, 32},
    {ExecutionMode::SW_FP32, HWGeneration::UNDEFINED, {1, 9}, 64},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_0, {1, 2}, 64},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_0, {1, 32}, 64},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_0, {1, 33}, 128},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_5, {1, 2}, 64},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_5, {1, 32}, 64},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_5, {1, 33}, 128},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_6, {1, 2}, 16},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_6, {1, 8}, 16},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_6, {1, 9}, 32},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_3_6, {1, 33}, 80},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_4_0, {1, 2}, 16},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_4_0, {1, 8}, 16},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_4_0, {1, 9}, 32},
    {ExecutionMode::SW_EXACT, HWGeneration::GNA_4_0, {1, 33}, 80},
};

class GNAPluginForMemoryAlignmentTest : public GNAPlugin {
public:
    GNAPluginForMemoryAlignmentTest(const std::map<std::string, std::string>& configMap) : GNAPlugin(configMap) {
        if (gnadevice) {
            gnamem.reset(new gna_memory_float(memory::GNAFloatAllocator{},
                                              gnadevice->getMemAlignment(),
                                              limitations::kMemoryPageSize));
            graphCompiler.setGNAMemoryPtr(gnamem);
            gnadevice.reset();
        }
    }

    size_t get_memory_REGION_INPUTS_size() {
        return this->gnamem->getQueue(ov::intel_gna::memory::REGION_INPUTS)->calcSize();
    }
};

class GNAPluginLoadNetworkMemAlignmentTests : public ::testing::TestWithParam<MemAlignmentTestParams> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<MemAlignmentTestParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "mode=" << param.execution_mode << "_";
        result << "target=" << param.compile_target << "_";
        result << "inp=" << param.input_shape.to_string() << "_";
        result << "inp_region_size=" << param.mem_input_region_size;
        return result.str();
    }

protected:
    void Run() {
        const ov::AnyMap gna_config = {ov::intel_gna::execution_mode(test_params.execution_mode),
                                       ov::intel_gna::compile_target(test_params.compile_target)};

        auto plugin = GNAPluginForMemoryAlignmentTest(any_copy(gna_config));
        auto function = getMulFunction(test_params.input_shape);
        CNNNetwork cnnNetwork(function);
        plugin.LoadNetwork(cnnNetwork);
        EXPECT_EQ(plugin.get_memory_REGION_INPUTS_size(), test_params.mem_input_region_size);
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

TEST_P(GNAPluginLoadNetworkMemAlignmentTests, checkInputRegionAlignment) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MemoryAlignment,
                         GNAPluginLoadNetworkMemAlignmentTests,
                         ::testing::ValuesIn(all_tests_params),
                         GNAPluginLoadNetworkMemAlignmentTests::GetTestCaseName);

class MemoryAlignmentTest : public ::testing::Test {};

TEST(MemoryAlignmentTest, getMemoryAlignmentBytes_ExpectExceptionWhenTargetIsUnset) {
    EXPECT_ANY_THROW(getMemoryAlignmentBytes(DeviceVersion::NotSet));
}

TEST(MemoryAlignmentTest, getMemoryAlignmentBytes_Expect64ByteAlignmentWhenTargetIsGNA3_0) {
    EXPECT_EQ(getMemoryAlignmentBytes(DeviceVersion::GNA3_0), 64);
}

TEST(MemoryAlignmentTest, getMemoryAlignmentBytes_Expect16ByteAlignmentWhenTargetIsGNA3_6) {
    EXPECT_EQ(getMemoryAlignmentBytes(DeviceVersion::GNA3_6), 16);
}

}  // namespace testing
