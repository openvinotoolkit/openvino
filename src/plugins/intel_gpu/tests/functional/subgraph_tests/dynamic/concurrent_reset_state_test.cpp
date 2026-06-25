// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for https://github.com/openvinotoolkit/openvino/issues/36458
// Validates that concurrent reset_state() on sibling InferRequests from the
// same CompiledModel does not corrupt shape inference in the GPU plugin.

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/llm_builders.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/infer_request.hpp"

#include <thread>
#include <atomic>
#include <vector>

namespace {

class ConcurrentResetStateTest : public ::testing::Test {
public:
    // Regression test for issue #36458:
    // Two InferRequests from the same CompiledModel run concurrently.
    // One thread does infer() while the other does reset_state().
    // Without the fix, reset_state() corrupts variable layouts mid-inference,
    // leading to shape validation failures like "() -> ()".
    void test_concurrent_reset_state_does_not_corrupt_inference() {
#if defined(ANDROID)
        GTEST_SKIP();
#endif
        auto core = ov::test::utils::PluginCache::get().core();

        ov::AnyMap properties = {
            ov::hint::inference_precision(ov::element::f16)
        };

        const size_t n_batch = 1;
        const size_t n_heads = 32;
        const size_t n_features = 10;
        const size_t context_size = 20;
        ov::element::Type element_type = ov::element::f16;

        const bool stateful = true;

        auto model = ov::test::utils::make_llm_kv_cache_pattern(
            n_batch, n_heads, n_features, element_type,
            2,        // concat_axis
            stateful,
            false,    // fuse_cache_reorder
            stateful  // build_state_initializer
        );
        auto compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);

        auto input0 = model->get_parameters().at(0);
        auto input1 = model->get_parameters().at(1);

        auto ireq1 = compiled_model.create_infer_request();
        auto ireq2 = compiled_model.create_infer_request();

        // Prepare inputs for request 1 (prefill)
        auto ireq1_input0 = ov::test::utils::create_and_fill_tensor_real_distribution(
            element_type, {n_batch, context_size, n_heads, n_features}, -0.5f, 0.5f, 1);
        auto ireq1_input1 = ov::test::utils::create_and_fill_tensor_real_distribution(
            element_type, {n_batch, n_heads, context_size, context_size}, -0.5f, 0.5f, 1);
        ireq1.set_tensor(input0, ireq1_input0);
        ireq1.set_tensor(input1, ireq1_input1);

        // Prepare inputs for request 2 (slightly different shape to stress test)
        auto ireq2_input0 = ov::test::utils::create_and_fill_tensor_real_distribution(
            element_type, {n_batch, context_size + 1, n_heads, n_features}, -0.5f, 0.5f, 555);
        auto ireq2_input1 = ov::test::utils::create_and_fill_tensor_real_distribution(
            element_type, {n_batch, n_heads, context_size + 1, context_size + 1}, -0.5f, 0.5f, 555);
        ireq2.set_tensor(input0, ireq2_input0);
        ireq2.set_tensor(input1, ireq2_input1);

        // Warm up: do initial inference on both requests so states are populated
        ireq1.infer();
        ireq2.infer();

        // Stress test: concurrent infer() on one request and reset_state() on the other
        const size_t num_iterations = 50;
        std::atomic<bool> has_exception{false};
        std::string exception_message;
        std::mutex exception_mutex;

        auto record_exception = [&](const std::string& msg) {
            std::lock_guard<std::mutex> lk(exception_mutex);
            if (!has_exception.exchange(true)) {
                exception_message = msg;
            }
        };

        for (size_t iter = 0; iter < num_iterations && !has_exception; iter++) {
            // Thread 1: infer on request 1
            std::thread t1([&]() {
                try {
                    ireq1.infer();
                } catch (const std::exception& e) {
                    record_exception(std::string("infer() threw: ") + e.what());
                }
            });

            // Thread 2: reset_state on request 2 (shares CompiledModel with request 1)
            std::thread t2([&]() {
                try {
                    ireq2.reset_state();
                } catch (const std::exception& e) {
                    record_exception(std::string("reset_state() threw: ") + e.what());
                }
            });

            t1.join();
            t2.join();

            // Re-infer on request 2 after reset to verify it still works
            if (!has_exception) {
                try {
                    ireq2.infer();
                } catch (const std::exception& e) {
                    record_exception(std::string("post-reset infer() threw: ") + e.what());
                }
            }
        }

        ASSERT_FALSE(has_exception) << "Concurrent reset_state()/infer() failed: " << exception_message;
    }
};

TEST_F(ConcurrentResetStateTest, smoke_concurrent_reset_state_no_corruption) {
    this->test_concurrent_reset_state_does_not_corrupt_inference();
}

}  // namespace
