// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pre_post_process/converter_factory.hpp"

#include <gtest/gtest.h>
#include <ie_system_conf.h>

using namespace ov::intel_gna::pre_post_processing;

#ifdef HAVE_AVX2
TEST(ConverterFactoryTests, TestAvx2Supported) {
    // if compiled with AVX2 support, must return valid converter when AVX2 available at runtime
    if (InferenceEngine::with_cpu_x86_avx2()) {
        EXPECT_NE(ConverterFactory::create_converter(), nullptr);
    } else {
        EXPECT_EQ(ConverterFactory::create_converter(), nullptr);
    }
}
#else
TEST(ConverterFactoryTests, TestConverterFactoryReturnsNullptr) {
    // if compiled without AVX2 support, ConverterFactory must return nullptr
    EXPECT_EQ(ConverterFactory::create_converter(), nullptr);
}
#endif  // HAVE_AVX2