// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils/kernel_cache_frontend.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace cldnn;

namespace {

kernels_cache::kernels_code make_pending_kernel(kernel_language language, std::string payload, std::string options, bool batch_compilation) {
    auto kernel = std::make_shared<kernel_string>();
    kernel->language = language;
    kernel->str = std::move(payload);
    kernel->options = std::move(options);
    kernel->entry_point = "main";
    kernel->batch_compilation = batch_compilation;

    kernel_impl_params params;
    kernels_cache::kernels_code pending;
    pending.emplace(params, kernels_cache::kernel_code({std::move(kernel)}, params, false));
    return pending;
}

kernel_cache_frontend_context make_context(const std::map<std::string, std::string>& headers) {
    constexpr size_t test_batch_limit = 8;
    constexpr uint32_t test_program_id = 17;
    kernel_cache_frontend_context context;
    context.max_kernels_per_batch = test_batch_limit;
    context.program_id = test_program_id;
    context.device_name = "portable-device";
    context.driver_version = "portable-driver";
    context.batch_headers = &headers;
    return context;
}

}  // namespace

TEST(kernel_cache_frontend, precompiled_spirv_bypasses_source_processing) {
    constexpr char spirv_bytes[] = "\x03\x02\x23\x07\0binary";
    const std::string spirv_payload(spirv_bytes, sizeof(spirv_bytes) - 1);
    const std::string options = "-opaque-option";
    auto pending = make_pending_kernel(kernel_language::SPIRV, spirv_payload, options, false);
    const std::map<std::string, std::string> headers{{"source-only-header", "must-not-touch-spirv"}};
    const auto context = make_context(headers);
    std::vector<kernels_cache::batch_program> batches;

    kernel_cache_frontend::prepare(pending, context, batches);

    ASSERT_EQ(batches.size(), 1u);
    EXPECT_EQ(batches[0].language, kernel_language::SPIRV);
    EXPECT_EQ(batches[0].source, kernels_cache::source_code{spirv_payload});
    EXPECT_EQ(batches[0].options, options);

    const auto cache_key = options + " __PROGRAM__0 __LANG__" + std::to_string(static_cast<size_t>(kernel_language::SPIRV));
    const auto hash_input = cache_key + " " + context.driver_version + context.device_name + spirv_payload;
    EXPECT_EQ(batches[0].hash_value, std::hash<std::string>()(hash_input));
}

TEST(kernel_cache_frontend, source_frontend_preserves_batch_option_normalization) {
    const std::string options = "-DSECOND=2 -DFIRST=1";
    auto pending = make_pending_kernel(kernel_language::OCLC, "kernel void main() {}", options, true);
    const std::map<std::string, std::string> headers;
    const auto context = make_context(headers);
    std::vector<kernels_cache::batch_program> batches;

    kernel_cache_frontend::prepare(pending, context, batches);

    ASSERT_EQ(batches.size(), 1u);
    EXPECT_EQ(batches[0].language, kernel_language::OCLC);
    EXPECT_EQ(batches[0].options, "-DFIRST=1 -DSECOND=2 ");
}

TEST(kernel_cache_frontend, referenced_headers_keep_common_preamble_outside_conditional_includes) {
    auto pending = make_pending_kernel(kernel_language::OCLC,
                                       "#if OPTIONAL_FEATURE\n#include \"common.cl\"\n#endif\n#include \"helper.cl\"\n",
                                       "-DOPTIONAL_FEATURE=0",
                                       false);
    const std::map<std::string, std::string> headers{
        {"common", "#define CAT_IMPL(lhs, rhs) lhs ## rhs\n#define CAT(lhs, rhs) CAT_IMPL(lhs, rhs)"},
        {"helper", "#include \"common.cl\"\nint CAT(helper, _value);"},
    };
    auto context = make_context(headers);
    context.source_headers = KernelSourceHeaders::REFERENCED_ONLY;
    std::vector<kernels_cache::batch_program> batches;

    kernel_cache_frontend::prepare(pending, context, batches);

    ASSERT_EQ(batches.size(), 1u);
    ASSERT_EQ(batches[0].source.size(), 2u);
    EXPECT_EQ(batches[0].source[0], headers.at("common") + "\n");
    EXPECT_EQ(batches[0].source[1].find("#include"), std::string::npos);
    EXPECT_NE(batches[0].source[1].find("int CAT(helper, _value);"), std::string::npos);
}
