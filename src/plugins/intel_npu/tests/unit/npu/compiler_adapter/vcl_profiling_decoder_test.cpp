// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_profiling_decoder.hpp"

#include <gtest/gtest.h>
#include <ze_graph_profiling_ext.h>

#include <memory>
#include <string>
#include <vector>

#include "fake_vcl.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

using ::fake_vcl::FakeVcl;
using ::intel_npu::VCLProfilingDecoder;

namespace {

ov::Tensor makeNetworkTensor() {
    return ov::Tensor(ov::element::u8, ov::Shape{4});
}

}  // namespace

struct VCLProfilingDecoderTest : public ::testing::Test {
    FakeVcl fake;

    std::shared_ptr<VCLProfilingDecoder> makeDecoder() {
        return std::make_shared<VCLProfilingDecoder>(fake.functions());
    }
};

TEST_F(VCLProfilingDecoderTest, ConstructionDoesNotTriggerACompilerCreateGetDestroyCycle) {
    // The whole point of the decoder: decoding never needs a compiler instance.
    auto decoder = makeDecoder();
    ASSERT_NE(decoder, nullptr);

    EXPECT_FALSE(fake.called("vclCompilerCreate"));
    EXPECT_FALSE(fake.called("vclCompilerGetProperties"));
    EXPECT_EQ(fake.compilerDestroyCount, 0);
}

TEST_F(VCLProfilingDecoderTest, NullFunctionTableIsRejected) {
    EXPECT_THROW(
        {
            auto decoder = std::make_shared<VCLProfilingDecoder>(nullptr);
            (void)decoder;
        },
        ov::Exception);
}

TEST_F(VCLProfilingDecoderTest, UnwiredFunctionTableIsRejected) {
    // A default-constructed table: every entry point is null.
    auto unwired = std::make_shared<const intel_npu::VCLFunctionTable>();
    EXPECT_THROW(
        {
            auto decoder = std::make_shared<VCLProfilingDecoder>(unwired);
            (void)decoder;
        },
        ov::Exception);
}

TEST_F(VCLProfilingDecoderTest, MissingAProfilingEntryPointIsRejected) {
    fake.mutableFunctions()->vclGetDecodedProfilingBuffer = nullptr;
    EXPECT_THROW(makeDecoder(), ov::Exception);
}

TEST_F(VCLProfilingDecoderTest, UnrelatedMissingEntryPointsDoNotBlockConstruction) {
    // The decoder only needs the profiling entry points, never the compiler/executable/query ones.
    fake.mutableFunctions()->vclCompilerCreate = nullptr;
    fake.mutableFunctions()->vclAllocatedExecutableCreate4 = nullptr;
    fake.mutableFunctions()->vclQueryNetworkCreate = nullptr;
    EXPECT_NO_THROW(makeDecoder());
}

TEST_F(VCLProfilingDecoderTest, DecodeFollowsCreateGetDestroyOrdering) {
    fake.profilingPayload.assign(2 * sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();

    const std::vector<uint8_t> profData{1, 2, 3};
    const ov::Tensor network = makeNetworkTensor();
    const auto info = decoder->decode(profData, network);

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

TEST_F(VCLProfilingDecoderTest, DecodeSizesByLayerInfoStride) {
    fake.profilingPayload.assign(5 * sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();
    EXPECT_EQ(decoder->decode({1}, makeNetworkTensor()).size(), 5u);
}

TEST_F(VCLProfilingDecoderTest, DecodeThrowsOnNullData) {
    fake.forceNullProfilingData = true;
    auto decoder = makeDecoder();

    try {
        decoder->decode({1}, makeNetworkTensor());
        FAIL() << "Expected a throw on NULL profiling data";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Failed to get VCL profiling output"), std::string::npos);
    }
}

TEST_F(VCLProfilingDecoderTest, DecodeThrowsWhenCreateFails) {
    // A decodable payload, so the scripted vclProfilingCreate() failure is the only reason to throw.
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();
    fake.failWith("vclProfilingCreate", VCL_RESULT_ERROR_UNKNOWN);

    try {
        decoder->decode({1}, makeNetworkTensor());
        FAIL() << "Expected a throw on vclProfilingCreate failure";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("vclProfilingCreate"), std::string::npos);
    }
    // Nothing was created, so nothing must be destroyed.
    EXPECT_EQ(fake.profilingDestroyCount, 0);
}

TEST_F(VCLProfilingDecoderTest, DecodeThrowsWhenDestroyFails) {
    // The payload must decode successfully, otherwise the null-data guard throws first and the
    // destroy-failure path is never reached.
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();
    fake.failWith("vclProfilingDestroy", VCL_RESULT_ERROR_UNKNOWN);

    try {
        decoder->decode({1}, makeNetworkTensor());
        FAIL() << "Expected a throw on vclProfilingDestroy failure";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("vclProfilingDestroy"), std::string::npos);
    }
    // The decode ran to completion before destroy was attempted.
    EXPECT_EQ(fake.callCount("vclGetDecodedProfilingBuffer"), 1u);
    EXPECT_EQ(fake.profilingDestroyCount, 1);
}
