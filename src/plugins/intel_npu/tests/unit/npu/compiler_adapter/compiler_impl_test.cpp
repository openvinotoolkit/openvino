// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fake_vcl.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "model_serializer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ze_graph_ext_wrappers.hpp"

using ::fake_vcl::FakeVcl;
using ::intel_npu::FilteredConfig;
using ::intel_npu::IDevice;
using ::intel_npu::OptionsDesc;
using ::intel_npu::VCLCompilerImpl;

namespace {

/// Registers just the options the compiler-in-plugin path reads, so `config.get<>` resolves.
std::shared_ptr<OptionsDesc> makeOptionsDesc() {
    auto desc = std::make_shared<OptionsDesc>();
    desc->add<::intel_npu::MODEL_SERIALIZER_VERSION>();
    desc->add<::intel_npu::WS_COMPILE_CALL_NUMBER>();
    return desc;
}

FilteredConfig makeConfig() {
    FilteredConfig config(makeOptionsDesc());
    // compileWsIterative writes this option, and FilteredConfig::update refuses keys that are
    // registered but not enabled. MODEL_SERIALIZER_VERSION is deliberately left disabled, matching a
    // configuration where the compiler does not advertise it.
    config.enable(std::string(ov::intel_npu::ws_compile_call_number.name()), true);
    return config;
}

/// A minimal model with one weight, enough for the serializer to produce a real IR.
std::shared_ptr<ov::Model> makeModel() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5});
    input->set_friendly_name("Parameter_0");
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);
    add->set_friendly_name("Add_1");
    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "compiler_impl_test_model");
}

struct VCLCompilerImplTest : public ::testing::Test {
    FakeVcl fake;

    std::shared_ptr<VCLCompilerImpl> makeCompiler(
        const std::optional<IDevice::DeviceProperties>& props = std::nullopt) {
        return std::make_shared<VCLCompilerImpl>(fake.api(), props);
    }
};

//
// --- construction ---
//

TEST_F(VCLCompilerImplTest, ConstructionQueriesVersionCreatesCompilerAndReadsProperties) {
    auto compiler = makeCompiler();
    ASSERT_NE(compiler, nullptr);

    EXPECT_TRUE(fake.called("vclGetVersion"));
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclCompilerGetProperties"), 1u);
    // vclGetVersion must precede compiler creation: the negotiated version goes into the desc.
    EXPECT_LT(fake.indexOf("vclGetVersion"), fake.indexOf("vclCompilerCreate"));
}

TEST_F(VCLCompilerImplTest, ConstructionForwardsTheNegotiatedVersionInTheCompilerDesc) {
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR};
    auto compiler = makeCompiler();

    ASSERT_EQ(fake.compilerDescs.size(), 1u);
    EXPECT_EQ(fake.compilerDescs[0].version.major, VCL_COMPILER_VERSION_MAJOR);
    EXPECT_EQ(fake.compilerDescs[0].version.minor, VCL_COMPILER_VERSION_MINOR);
}

TEST_F(VCLCompilerImplTest, DevicePropertiesAreForwardedExactly) {
    IDevice::DeviceProperties props{0x1234u, 7u, 4u};
    auto compiler = makeCompiler(props);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    const auto& desc = fake.deviceDescs[0];
    EXPECT_EQ(desc.size, sizeof(vcl_device_desc_t));
    EXPECT_EQ(desc.deviceID, 0x1234u);
    EXPECT_EQ(desc.revision, 7u);
    EXPECT_EQ(desc.tileCount, 4u);
}

TEST_F(VCLCompilerImplTest, OversizedSubdeviceIdClampsToTheInvalidRevisionSentinel) {
    constexpr auto sentinel = std::numeric_limits<uint16_t>::max();
    // Anything at or above the sentinel cannot fit the 16-bit revision field.
    IDevice::DeviceProperties props{0x1u, static_cast<uint32_t>(sentinel) + 5u, 1u};
    auto compiler = makeCompiler(props);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    EXPECT_EQ(fake.deviceDescs[0].revision, sentinel);
}

TEST_F(VCLCompilerImplTest, SubdeviceIdExactlyAtTheSentinelAlsoClamps) {
    constexpr auto sentinel = std::numeric_limits<uint16_t>::max();
    IDevice::DeviceProperties props{0x1u, static_cast<uint32_t>(sentinel), 1u};
    auto compiler = makeCompiler(props);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    EXPECT_EQ(fake.deviceDescs[0].revision, sentinel);
}

TEST_F(VCLCompilerImplTest, AbsentDevicePropertiesUseDefaultSentinels) {
    auto compiler = makeCompiler(std::nullopt);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    const auto& desc = fake.deviceDescs[0];
    EXPECT_EQ(desc.size, sizeof(vcl_device_desc_t));
    EXPECT_EQ(desc.deviceID, 0x00u);
    EXPECT_EQ(desc.revision, std::numeric_limits<uint16_t>::max());
    EXPECT_EQ(desc.tileCount, std::numeric_limits<uint32_t>::max());
}

TEST_F(VCLCompilerImplTest, NullApiTableIsRejected) {
    EXPECT_THROW(
        {
            auto compiler = std::make_shared<VCLCompilerImpl>(nullptr);
            (void)compiler;
        },
        ov::Exception);
}

TEST_F(VCLCompilerImplTest, GetVersionFailureThrows) {
    fake.failWith("vclGetVersion", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(makeCompiler(), ov::Exception);
}

TEST_F(VCLCompilerImplTest, CompilerCreateFailureThrowsWithTheVclLogAppended) {
    fake.logString = "compiler-create-exploded";
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_OUT_OF_MEMORY);

    try {
        makeCompiler();
        FAIL() << "Expected construction to throw";
    } catch (const ov::Exception& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("vclCompilerCreate"), std::string::npos) << what;
        EXPECT_NE(what.find("compiler-create-exploded"), std::string::npos) << what;
    }
}

TEST_F(VCLCompilerImplTest, CompilerGetPropertiesFailureThrows) {
    fake.failWith("vclCompilerGetProperties", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(makeCompiler(), ov::Exception);
}

TEST_F(VCLCompilerImplTest, ConstructionAcceptsALibraryBelowTheFloorAndDefersTheCheck) {
    // Documents where the version gate actually lives: construction succeeds even for a too-old
    // library, and the refusal happens on first compile/query (see CompileThrowsWhenTheLibraryIs...).
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR - 1, 0};
    auto compiler = makeCompiler();

    ASSERT_NE(compiler, nullptr);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

//
// --- destruction ---
//

TEST_F(VCLCompilerImplTest, DestructionDestroysTheCompilerExactlyOnce) {
    {
        auto compiler = makeCompiler();
    }
    EXPECT_EQ(fake.compilerDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, DestructionSwallowsCompilerDestroyFailure) {
    fake.failWith("vclCompilerDestroy", VCL_RESULT_ERROR_UNKNOWN);
    // A throwing destructor would terminate; the implementation only warns.
    EXPECT_NO_THROW({ auto compiler = makeCompiler(); });
    EXPECT_EQ(fake.compilerDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, FailedConstructionDoesNotDestroyAnUncreatedCompiler) {
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(makeCompiler(), ov::Exception);
    EXPECT_EQ(fake.compilerDestroyCount, 0);
}

//
// --- get_version ---
//

TEST_F(VCLCompilerImplTest, GetVersionPacksTheReportedPropertiesVersion) {
    fake.propertiesVersion = {9, 3};
    auto compiler = makeCompiler();

    EXPECT_EQ(compiler->get_version(), ZE_MAKE_VERSION(9, 3));
    EXPECT_EQ(compiler->get_version() >> 16, 9u);
    EXPECT_EQ(compiler->get_version() & 0xFFFFu, 3u);
}

//
// --- is_option_supported ---
//

TEST_F(VCLCompilerImplTest, IsOptionSupportedReturnsTrueOnSuccess) {
    auto compiler = makeCompiler();
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedPassesNullptrWhenNoValueIsGiven) {
    auto compiler = makeCompiler();
    compiler->is_option_supported("SOME_OPTION");

    ASSERT_EQ(fake.optionSupportQueries.size(), 1u);
    EXPECT_EQ(fake.optionSupportQueries[0].first, "SOME_OPTION");
    EXPECT_FALSE(fake.optionSupportQueries[0].second.has_value());
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedForwardsTheValueWhenGiven) {
    auto compiler = makeCompiler();
    compiler->is_option_supported("SOME_OPTION", std::string("SOME_VALUE"));

    ASSERT_EQ(fake.optionSupportQueries.size(), 1u);
    EXPECT_EQ(fake.optionSupportQueries[0].first, "SOME_OPTION");
    ASSERT_TRUE(fake.optionSupportQueries[0].second.has_value());
    EXPECT_EQ(*fake.optionSupportQueries[0].second, "SOME_VALUE");
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedSwallowsErrorsAndReportsFalse) {
    auto compiler = makeCompiler();
    // The exception is deliberately swallowed: older libraries lack this entry point.
    fake.failWith("vclGetCompilerIsOptionSupported", VCL_RESULT_ERROR_UNSUPPORTED_FEATURE);
    EXPECT_FALSE(compiler->is_option_supported("SOME_OPTION"));

    fake.failWith("vclGetCompilerIsOptionSupported", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_FALSE(compiler->is_option_supported("SOME_OPTION"));
}

//
// --- get_supported_options ---
//

TEST_F(VCLCompilerImplTest, GetSupportedOptionsUsesTheTwoCallSizeProtocol) {
    auto compiler = makeCompiler();
    const auto options = compiler->get_supported_options();

    EXPECT_EQ(fake.callCount("vclGetCompilerSupportedOptions"), 2u);
    EXPECT_EQ(options, std::vector<std::string>({"OPT_A", "OPT_B", "OPT_C"}));
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsReturnsEmptyOnZeroSizeWithoutASecondCall) {
    fake.supportedOptionsBuffer.clear();
    auto compiler = makeCompiler();
    const auto options = compiler->get_supported_options();

    EXPECT_TRUE(options.empty());
    // Size 0 short-circuits: only the sizing call happens.
    EXPECT_EQ(fake.callCount("vclGetCompilerSupportedOptions"), 1u);
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsTrimsTrailingNulsAndTokenises) {
    // VCL pads the buffer with NULs; they must not become part of an option name.
    fake.supportedOptionsBuffer = std::string("OPT_A OPT_B") + std::string(5, '\0');
    auto compiler = makeCompiler();
    const auto options = compiler->get_supported_options();

    ASSERT_EQ(options.size(), 2u);
    EXPECT_EQ(options[0], "OPT_A");
    EXPECT_EQ(options[1], "OPT_B");
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsCollapsesArbitraryWhitespace) {
    fake.supportedOptionsBuffer = "  OPT_A \t OPT_B \n OPT_C  ";
    auto compiler = makeCompiler();
    EXPECT_EQ(compiler->get_supported_options(), std::vector<std::string>({"OPT_A", "OPT_B", "OPT_C"}));
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsReturnsEmptyWhenTheBufferIsAllNuls) {
    fake.supportedOptionsBuffer = std::string(8, '\0');
    auto compiler = makeCompiler();
    EXPECT_TRUE(compiler->get_supported_options().empty());
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsThrowsWhenTheSizingCallFails) {
    auto compiler = makeCompiler();
    fake.failWith("vclGetCompilerSupportedOptions", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(compiler->get_supported_options(), ov::Exception);
}

//
// --- process_profiling_output ---
//

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputFollowsCreateGetDestroyOrdering) {
    fake.profilingPayload.assign(2 * sizeof(ze_profiling_layer_info), 0);
    auto compiler = makeCompiler();

    const std::vector<uint8_t> profData{1, 2, 3};
    const std::vector<uint8_t> network{4, 5, 6, 7};
    const auto info = compiler->process_profiling_output(profData, network);

    EXPECT_EQ(fake.callCount("vclProfilingCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclProfilingGetProperties"), 1u);
    EXPECT_EQ(fake.callCount("vclGetDecodedProfilingBuffer"), 1u);
    EXPECT_EQ(fake.profilingDestroyCount, 1);

    EXPECT_LT(fake.indexOf("vclProfilingCreate"), fake.indexOf("vclProfilingGetProperties"));
    EXPECT_LT(fake.indexOf("vclProfilingGetProperties"), fake.indexOf("vclGetDecodedProfilingBuffer"));
    EXPECT_LT(fake.indexOf("vclGetDecodedProfilingBuffer"), fake.indexOf("vclProfilingDestroy"));

    // One entry per ze_profiling_layer_info in the returned buffer.
    EXPECT_EQ(info.size(), 2u);
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputSizesByLayerInfoStride) {
    fake.profilingPayload.assign(5 * sizeof(ze_profiling_layer_info), 0);
    auto compiler = makeCompiler();
    EXPECT_EQ(compiler->process_profiling_output({1}, {2}).size(), 5u);
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputThrowsOnNullData) {
    fake.forceNullProfilingData = true;
    auto compiler = makeCompiler();

    try {
        compiler->process_profiling_output({1}, {2});
        FAIL() << "Expected a throw on NULL profiling data";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Failed to get VCL profiling output"), std::string::npos);
    }
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputThrowsWhenCreateFails) {
    auto compiler = makeCompiler();
    fake.failWith("vclProfilingCreate", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(compiler->process_profiling_output({1}, {2}), ov::Exception);
    EXPECT_EQ(fake.profilingDestroyCount, 0);
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputThrowsWhenDestroyFails) {
    auto compiler = makeCompiler();
    fake.failWith("vclProfilingDestroy", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(compiler->process_profiling_output({1}, {2}), ov::Exception);
}

//
// --- compile ---
//

TEST_F(VCLCompilerImplTest, CompileProducesABlobAndACompatibilityString) {
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);

    EXPECT_EQ(fake.callCount("vclAllocatedExecutableCreate4"), 1u);
    EXPECT_GT(tensor.get_byte_size(), 0u);
    ASSERT_TRUE(compatibility.has_value());
    EXPECT_EQ(*compatibility, "fake-compat");
}

TEST_F(VCLCompilerImplTest, CompileBuildFlagsAreIoInfoThenSpaceThenSerializedConfig) {
    // This is the plugin's actual contract with the compiler; nothing else pins it down.
    auto compiler = makeCompiler();
    auto config = makeConfig();
    const auto model = makeModel();

    const auto [tensor, compatibility] = compiler->compile(model, config);
    (void)tensor;
    (void)compatibility;

    ze_graph_compiler_version_info_t compilerVersion{};
    compilerVersion.major = fake.propertiesVersion.major;
    compilerVersion.minor = fake.propertiesVersion.minor;

    const auto isSupported = [&compiler](const std::string& name) {
        return compiler->is_option_supported(name);
    };
    const std::string expected = ::intel_npu::compiler_utils::serializeIOInfo(model, true) + " " +
                                 ::intel_npu::compiler_utils::serializeConfig(config, compilerVersion, isSupported);

    ASSERT_EQ(fake.buildFlags.size(), 1u);
    EXPECT_EQ(fake.buildFlags[0], expected);
}

TEST_F(VCLCompilerImplTest, CompileReturnsTheAlignedAllocatorSizeNotTheVclBlobSize) {
    // VCL reports the logical blob size; the tensor must expose the page-aligned allocation, because
    // that is what was actually reserved and what the deleter will free.
    fake.blobPayload.assign(5, 0xAB);
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);
    (void)compatibility;

    const size_t alignedExpected = ::intel_npu::utils::align_size_to_standard_page_size(5);
    EXPECT_EQ(tensor.get_byte_size(), alignedExpected);
    EXPECT_NE(tensor.get_byte_size(), 5u);
    // The payload still lands at the start of the buffer.
    EXPECT_EQ(static_cast<const uint8_t*>(tensor.data())[0], 0xAB);
}

TEST_F(VCLCompilerImplTest, CompileDestroysTheExecutableOnTheSuccessPath) {
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto result = compiler->compile(makeModel(), config);
    (void)result;

    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileCleansUpAllocationsWhenExecutableCreationFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.logString = "create4-failed";
    fake.failWith("vclAllocatedExecutableCreate4", VCL_RESULT_ERROR_OUT_OF_MEMORY);

    try {
        compiler->compile(makeModel(), config);
        FAIL() << "Expected compile to throw";
    } catch (const ov::Exception& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("vclAllocatedExecutableCreate4"), std::string::npos) << what;
        EXPECT_NE(what.find("create4-failed"), std::string::npos) << what;
    }
}

TEST_F(VCLCompilerImplTest, CompileDestroysTheExecutableWhenTheCompatibilityLookupThrows) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    // A hard failure (not UNSUPPORTED_FEATURE) propagates, but must not leak the executable.
    fake.failWith("vclExecutableGetCompatibilityString", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileTreatsUnsupportedCompatibilityStringAsAbsentNotAnError) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.compatibilityString.reset();  // makes the fake report UNSUPPORTED_FEATURE

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);
    (void)tensor;

    EXPECT_FALSE(compatibility.has_value());
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileTrimsTheTrailingNulFromTheCompatibilityString) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.compatibilityString = std::string("compat-value");

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);
    (void)tensor;

    ASSERT_TRUE(compatibility.has_value());
    // No embedded NUL should survive into the string.
    EXPECT_EQ(*compatibility, "compat-value");
    EXPECT_EQ(compatibility->size(), std::string("compat-value").size());
}

TEST_F(VCLCompilerImplTest, CompileThrowsOnZeroSizedCompatibilityString) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.compatibilityString = std::string("ignored");
    fake.compatibilityStringSizeOverride = 0u;

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileThrowsWhenExecutableDestroyFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclExecutableDestroy", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
}

TEST_F(VCLCompilerImplTest, CompileThrowsWhenTheLibraryIsBelowTheSupportedFloor) {
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR - 1, 0};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
    // The version gate fires before any executable is created.
    EXPECT_FALSE(fake.called("vclAllocatedExecutableCreate4"));
}

//
// --- compileWsOneShot ---
//

TEST_F(VCLCompilerImplTest, CompileWsOneShotReturnsOneTensorPerAllocation) {
    fake.wsBlobSizes = {8, 16, 32};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensors, compatibility] = compiler->compileWsOneShot(makeModel(), config);

    EXPECT_EQ(fake.callCount("vclAllocatedExecutableCreateWSOneShot2"), 1u);
    ASSERT_EQ(tensors.size(), 3u);
    ASSERT_TRUE(compatibility.has_value());
    EXPECT_EQ(*compatibility, "fake-compat");
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotOrdersInitSchedulesBeforeMain) {
    // The adapter consumes the last tensor as the main schedule, so allocation order is load-bearing.
    fake.wsBlobSizes = {8, 16, 4096 * 3};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensors, compatibility] = compiler->compileWsOneShot(makeModel(), config);
    (void)compatibility;

    ASSERT_EQ(tensors.size(), 3u);
    EXPECT_EQ(tensors.back().get_byte_size(), ::intel_npu::utils::align_size_to_standard_page_size(4096 * 3));
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotTensorsRemainValidAfterTheCompilerIsGone) {
    // m_info is cleared so ownership transfers to the tensors; a double free would show up here.
    fake.wsBlobSizes = {8, 16};
    std::vector<ov::Tensor> tensors;
    {
        auto compiler = makeCompiler();
        auto config = makeConfig();
        auto result = compiler->compileWsOneShot(makeModel(), config);
        tensors = std::move(result.first);
    }
    ASSERT_EQ(tensors.size(), 2u);
    for (auto& tensor : tensors) {
        ASSERT_NE(tensor.data(), nullptr);
        // Touch every byte: a freed buffer would trip the allocator or a sanitizer here.
        std::memset(tensor.data(), 0x5A, tensor.get_byte_size());
    }
    tensors.clear();
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotThrowsAndDestroysExecutableWhenNothingWasAllocated) {
    fake.wsBlobSizes.clear();  // no allocations -> m_info stays empty
    auto compiler = makeCompiler();
    auto config = makeConfig();

    try {
        compiler->compileWsOneShot(makeModel(), config);
        FAIL() << "Expected compileWsOneShot to throw";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("blobCount is zero"), std::string::npos) << error.what();
    }
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotThrowsWhenCreationFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclAllocatedExecutableCreateWSOneShot2", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->compileWsOneShot(makeModel(), config), ov::Exception);
}

//
// --- compileWsIterative ---
//

TEST_F(VCLCompilerImplTest, CompileWsIterativeGoesThroughTheSingleBlobPath) {
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensor, compatibility] = compiler->compileWsIterative(makeModel(), config, 2);
    (void)compatibility;

    EXPECT_EQ(fake.callCount("vclAllocatedExecutableCreate4"), 1u);
    EXPECT_GT(tensor.get_byte_size(), 0u);
}

//
// --- query ---
//

TEST_F(VCLCompilerImplTest, QueryParsesSupportedLayersAndKeysThemToNPU) {
    const std::string payload = "<Parameter_0><Add_1>";
    fake.queryResultBuffer.assign(payload.begin(), payload.end());
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto supported = compiler->query(makeModel(), config);

    ASSERT_EQ(supported.size(), 2u);
    ASSERT_TRUE(supported.count("Parameter_0"));
    ASSERT_TRUE(supported.count("Add_1"));
    EXPECT_EQ(supported.at("Parameter_0"), "NPU");
    EXPECT_EQ(supported.at("Add_1"), "NPU");
}

TEST_F(VCLCompilerImplTest, QueryUsesTheTwoCallSizeProtocolAndDestroysTheHandle) {
    const std::string payload = "<Add_1>";
    fake.queryResultBuffer.assign(payload.begin(), payload.end());
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto supported = compiler->query(makeModel(), config);
    (void)supported;

    EXPECT_EQ(fake.callCount("vclQueryNetworkCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclQueryNetwork"), 2u);
    EXPECT_EQ(fake.queryDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, QueryReturnsAnEmptyMapForAnEmptyResult) {
    fake.queryResultBuffer.clear();
    auto compiler = makeCompiler();
    auto config = makeConfig();

    EXPECT_TRUE(compiler->query(makeModel(), config).empty());
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenCreationFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclQueryNetworkCreate", VCL_RESULT_ERROR_INVALID_IR);

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
    EXPECT_EQ(fake.queryDestroyCount, 0);
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenTheResultFetchFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclQueryNetwork", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenDestroyFails) {
    const std::string payload = "<Add_1>";
    fake.queryResultBuffer.assign(payload.begin(), payload.end());
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclQueryNetworkDestroy", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
}

}  // namespace
