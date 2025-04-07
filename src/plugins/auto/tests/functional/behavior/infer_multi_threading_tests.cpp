// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <future>

#include "auto_func_test.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, can_run_3syncrequests_consistently_from_threads) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(
                           model_can_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    auto f1 = std::async(std::launch::async, [&] {
        req1.infer();
    });
    auto f2 = std::async(std::launch::async, [&] {
        req2.infer();
    });
    auto f3 = std::async(std::launch::async, [&] {
        req3.infer();
    });

    f1.wait();
    f2.wait();
    f3.wait();

    OV_ASSERT_NO_THROW(f1.get());
    OV_ASSERT_NO_THROW(f2.get());
    OV_ASSERT_NO_THROW(f3.get());
}

TEST_F(AutoFuncTests, can_run_3asyncrequests_consistently_from_threads_without_wait) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(
                           model_can_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req1.infer());
    OV_ASSERT_NO_THROW(req2.infer());
    OV_ASSERT_NO_THROW(req3.infer());

    auto f1 = std::async(std::launch::async, [&] {
        req1.start_async();
    });
    auto f2 = std::async(std::launch::async, [&] {
        req2.start_async();
    });
    auto f3 = std::async(std::launch::async, [&] {
        req3.start_async();
    });

    f1.wait();
    f2.wait();
    f3.wait();

    OV_ASSERT_NO_THROW(f1.get());
    OV_ASSERT_NO_THROW(f2.get());
    OV_ASSERT_NO_THROW(f3.get());
}

TEST_F(AutoFuncTests, can_run_3asyncrequests_consistently_with_wait) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(
                           model_can_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    req1.start_async();
    OV_ASSERT_NO_THROW(req1.wait());

    req2.start_async();
    OV_ASSERT_NO_THROW(req2.wait());

    req3.start_async();
    OV_ASSERT_NO_THROW(req3.wait());
}

TEST_F(AutoFuncTests, can_run_3asyncrequests_parallel_with_wait) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(
                           model_can_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    req1.start_async();
    req2.start_async();
    req3.start_async();

    OV_ASSERT_NO_THROW(req2.wait());
    OV_ASSERT_NO_THROW(req1.wait());
    OV_ASSERT_NO_THROW(req3.wait());
}
