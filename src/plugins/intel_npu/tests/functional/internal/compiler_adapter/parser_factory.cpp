// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <exception>
#include <memory>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/except.hpp"
#include "vcl_provider_utils.hpp"

namespace {

using namespace ::intel_npu;
namespace vcl_provider = ov::test::npu::vcl_provider;

class ParserFactoryFunctionalTests : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            initStructs = ZeroInitStructsHolder::getInstance();
        } catch (const std::exception& ex) {
            GTEST_SKIP() << "Couldn't initialize the Level Zero structures: " << ex.what();
        }
    }

    std::shared_ptr<ZeroInitStructsHolder> initStructs;
    ParserFactory factory;
};

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingDisabledDoesNotInvokeProvider) {
    std::size_t calls = 0;
    const auto parser =
        factory.getParser(initStructs, vcl_provider::counted(calls, vcl_provider::processWide()), false);
    EXPECT_NE(parser, nullptr);
    EXPECT_EQ(calls, 0u);
}

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingEnabledWithUnavailableProviderStillReturnsParser) {
    std::size_t calls = 0;
    const auto parser = factory.getParser(initStructs, vcl_provider::counted(calls, vcl_provider::unavailable()), true);
    EXPECT_NE(parser, nullptr);
    EXPECT_EQ(calls, 1u);
}

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingEnabledWithoutProviderThrows) {
    EXPECT_THROW(factory.getParser(initStructs, VCLFunctionTableProvider{}, true), ov::Exception);
}

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingEnabledWithAvailableProviderInvokesProviderOnce) {
    try {
        vcl_provider::processWide()();
    } catch (const std::exception&) {
        GTEST_SKIP() << "Couldn't load compiler library";
    }

    std::size_t calls = 0;
    const auto parser = factory.getParser(initStructs, vcl_provider::counted(calls, vcl_provider::processWide()), true);
    EXPECT_NE(parser, nullptr);
    EXPECT_EQ(calls, 1u);
}

TEST_F(ParserFactoryFunctionalTests, GetParserWithoutInitStructsThrowsBeforeInvokingProvider) {
    EXPECT_THROW(
        factory.getParser(nullptr, vcl_provider::forbidden("The Level Zero structures are validated first"), true),
        ov::Exception);
}

}  // namespace
