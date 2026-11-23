// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/except.hpp"

namespace {

using namespace ::intel_npu;

class ParserFactoryFunctionalTests : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        initStructs = ZeroInitStructsHolder::getInstance();
        loader = std::make_shared<VCLLoader>();
    }

    // Hands out the real VCL entry points and counts how many times they were requested.
    VCLFunctionTableProvider countingProvider() {
        return [this] {
            ++calls;
            return loader->sharedFunctions();
        };
    }

    std::shared_ptr<ZeroInitStructsHolder> initStructs;
    std::shared_ptr<VCLLoader> loader;
    std::size_t calls = 0;
    ParserFactory factory;
};

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingDisabledDoesNotInvokeProvider) {
    const auto parser = factory.getParser(initStructs, countingProvider(), false);
    EXPECT_NE(parser, nullptr);
    EXPECT_EQ(calls, 0u);
}

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingEnabledWithUnavailableProviderStillReturnsParser) {
    const VCLFunctionTableProvider unavailable = [this]() -> std::shared_ptr<const VCLFunctionTable> {
        ++calls;
        OPENVINO_THROW("The VCL library cannot be loaded");
    };
    const auto parser = factory.getParser(initStructs, unavailable, true);
    EXPECT_NE(parser, nullptr);
    EXPECT_EQ(calls, 1u);
}

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingEnabledWithoutProviderStillReturnsParser) {
    const auto parser = factory.getParser(initStructs, VCLFunctionTableProvider{}, true);
    EXPECT_NE(parser, nullptr);
}

TEST_F(ParserFactoryFunctionalTests, GetParserProfilingEnabledInvokesProviderOnce) {
    const auto parser = factory.getParser(initStructs, countingProvider(), true);
    EXPECT_NE(parser, nullptr);
    EXPECT_EQ(calls, 1u);
}

TEST_F(ParserFactoryFunctionalTests, GetParserWithoutInitStructsThrowsBeforeInvokingProvider) {
    EXPECT_THROW(factory.getParser(nullptr, countingProvider(), true), ov::Exception);
    EXPECT_EQ(calls, 0u);
}

}  // namespace
