// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"
#include "vcl_provider_utils.hpp"

namespace {

using namespace ::intel_npu;
namespace vcl_provider = ov::test::npu::vcl_provider;

TEST(ParserFactoryTests, GetParser_WithoutInitStructs_ThrowsBeforeInvokingProvider) {
    const ParserFactory factory;
    EXPECT_THROW(
        factory.getParser(nullptr, vcl_provider::forbidden("The Level Zero structures are validated first"), true),
        ov::Exception);
}

}  // namespace
