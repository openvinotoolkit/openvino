// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#ifdef OV_GPU_WITH_SYCL_RT

#include "test_utils.h"

#include <exception>
#include <memory>

namespace sycl_tests {

struct sycl_test_context {
    std::shared_ptr<cldnn::engine> sycl_test_engine;
    std::shared_ptr<cldnn::stream> sycl_test_stream;
};

inline sycl_test_context create_sycl_test_context() {
    sycl_test_context ctx;
    try {
        ctx.sycl_test_engine = ::tests::create_test_engine(cldnn::engine_types::sycl, cldnn::runtime_types::sycl);
    } catch (const std::exception& e) {
        OPENVINO_THROW(e.what());
    }

    OPENVINO_ASSERT(ctx.sycl_test_engine != nullptr, "[GPU] Failed to create SYCL engine for tests");
    ctx.sycl_test_stream = ctx.sycl_test_engine->create_stream(::tests::get_test_default_config(*ctx.sycl_test_engine));
    return ctx;
}

}  // namespace sycl_tests

#endif  // OV_GPU_WITH_SYCL_RT
