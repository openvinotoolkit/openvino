// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/util/file_util.hpp"

// Builds a real model through the real CPU plugin (via ov::Core/register_plugin) instead of mocks,
// so CPUStreamsExecutor's actual task-queue/exception mechanics are genuinely exercised.
// ControllableOp is picked up by the CPU plugin's generic Reference-node fallback, needing no
// plugin-internal registration.

// ControllableOp is kept out of the anonymous namespace below: OPENVINO_OP()'s RTTI machinery
// trips a hidden-visibility warning/error on some compilers (e.g. RISC-V GCC) when its enclosing
// class has no linkage.

// Shared control block between a test and the ControllableOp instance(s) in its compiled model.
struct ControllableOpState {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> entered{false};
    std::atomic<bool> released{true};  // default: evaluate() does not block
    std::atomic<bool> should_throw{false};
    std::atomic<std::thread::id> executing_thread_id{};

    // Blocks the calling (test) thread until some evaluate() call has started.
    void wait_until_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] {
            return entered.load();
        });
    }

    // Lets a blocked evaluate() call proceed.
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        cv.notify_all();
    }

    // Re-arms gating for another controlled (blocking) evaluate() call.
    void arm_gate() {
        std::lock_guard<std::mutex> lock(mutex);
        entered = false;
        released = false;
    }
};

// Minimal custom op: copies its single input to its single output, optionally blocking until
// released and/or throwing, and records which thread actually executed it.
class ControllableOp : public ov::op::Op {
public:
    OPENVINO_OP("AsyncInferRequestTestControllableOp");

    ControllableOp() = default;
    ControllableOp(const ov::OutputVector& args, std::shared_ptr<ControllableOpState> state)
        : Op(args),
          m_state(std::move(state)) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
        return std::make_shared<ControllableOp>(new_args, m_state);
    }

    bool visit_attributes(ov::AttributeVisitor&) override {
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        m_state->executing_thread_id = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            m_state->entered = true;
        }
        m_state->cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(m_state->mutex);
            m_state->cv.wait(lock, [&] {
                return m_state->released.load();
            });
        }
        if (m_state->should_throw.load()) {
            throw std::runtime_error("ControllableOp: controlled evaluate failure");
        }
        std::memcpy(outputs[0].data(), inputs[0].data(), inputs[0].get_byte_size());
        return true;
    }

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  const ov::EvaluationContext&) const override {
        return evaluate(outputs, inputs);
    }

private:
    std::shared_ptr<ControllableOpState> m_state;
};

namespace {

// Device name for the test-local ov::Core's registered CPU plugin -- distinct from "CPU" so it
// never collides with ambient/default plugin registration.
constexpr const char* kTestDeviceName = "ASYNC_INFER_REQUEST_TEST_CPU";

// Lazily builds a process-local ov::Core with the real CPU plugin registered, reused across tests.
// A bare library name is passed to register_plugin() so the OS loader resolves it (needed since
// some CI layouts, e.g. Android, don't co-locate the plugin .so with the test binary); the base
// name varies by architecture (see CMakeLists.txt's per-arch OUTPUT_NAME).
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
constexpr const char* kCpuPluginBaseName = "openvino_arm_cpu_plugin";
#elif defined(OPENVINO_ARCH_RISCV64)
constexpr const char* kCpuPluginBaseName = "openvino_riscv_cpu_plugin";
#else
constexpr const char* kCpuPluginBaseName = "openvino_intel_cpu_plugin";
#endif

ov::Core& GetTestCore() {
    static ov::Core core = [] {
        ov::Core c;
        auto plugin_name = ov::util::make_plugin_library_name(std::string(kCpuPluginBaseName) + OV_BUILD_POSTFIX);
        c.register_plugin(plugin_name.string(), kTestDeviceName);
        return c;
    }();
    return core;
}

struct TestInferContext {
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::shared_ptr<ControllableOpState> state;
};

// Builds Parameter -> ControllableOp -> Result, compiles it with the given properties through the
// real CPU plugin (see GetTestCore()), and creates one real AsyncInferRequest on top of it.
TestInferContext CreateTestInferRequest(const ov::AnyMap& properties) {
    auto state = std::make_shared<ControllableOpState>();
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto op = std::make_shared<ControllableOp>(ov::OutputVector{param}, state);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "AsyncInferRequestTestModel");

    auto compiled_model = GetTestCore().compile_model(model, kTestDeviceName, properties);
    auto infer_request = compiled_model.create_infer_request();
    return {compiled_model, infer_request, state};
}

void SetInput(ov::InferRequest& request, const ov::Output<const ov::Node>& port, float value) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{1});
    *tensor.data<float>() = value;
    request.set_tensor(port, tensor);
}

float GetOutput(ov::InferRequest& request, const ov::Output<const ov::Node>& port) {
    return *request.get_tensor(port).data<float>();
}

// 1 stream, 1 thread: yields is_optimized_single_stream == true without requesting
// CPU_MULTI_APP_THREAD_SYNC_EXECUTION at all.
ov::AnyMap OptimizedSingleStreamProperties() {
    return {ov::num_streams(1), ov::inference_num_threads(1)};
}

// 4 streams with multi_app_thread_sync_execution=true (pinning off, since the plugin rejects that
// combination with pinning resolved to true).
ov::AnyMap ExplicitMultiAppThreadSyncProperties() {
    return {ov::num_streams(4),
            ov::inference_num_threads(4),
            ov::hint::enable_cpu_pinning(false),
            ov::intel_cpu::multi_app_thread_sync_execution(true)};
}

}  // namespace

TEST(AsyncInferRequestTest, UsesCallerThreadForOptimizedSingleStream) {
    auto ctx = CreateTestInferRequest(OptimizedSingleStreamProperties());
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 11.0F);

    const auto caller_thread_id = std::this_thread::get_id();
    ASSERT_NO_THROW(ctx.infer_request.infer());

    ASSERT_EQ(ctx.state->executing_thread_id.load(), caller_thread_id);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 11.0F);
}

TEST(AsyncInferRequestTest, UsesCallerThreadForExplicitMultiAppThreadSync) {
    auto ctx = CreateTestInferRequest(ExplicitMultiAppThreadSyncProperties());
    ASSERT_TRUE(ctx.compiled_model.get_property(ov::intel_cpu::multi_app_thread_sync_execution.name()).as<bool>());

    SetInput(ctx.infer_request, ctx.compiled_model.input(), 22.0F);
    const auto caller_thread_id = std::this_thread::get_id();
    ASSERT_NO_THROW(ctx.infer_request.infer());

    ASSERT_EQ(ctx.state->executing_thread_id.load(), caller_thread_id);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 22.0F);
}

TEST(AsyncInferRequestTest, KeepsExecutorDispatchForSyncInferWhenCallerThreadModeIsDisabled) {
    // Same stream count as ExplicitMultiAppThreadSyncProperties, but without the property set, so
    // dispatch must still go through the base class's worker-thread executor.
    auto ctx = CreateTestInferRequest({ov::num_streams(4), ov::inference_num_threads(4)});
    ASSERT_FALSE(ctx.compiled_model.get_property(ov::intel_cpu::multi_app_thread_sync_execution.name()).as<bool>());

    SetInput(ctx.infer_request, ctx.compiled_model.input(), 33.0F);
    const auto caller_thread_id = std::this_thread::get_id();
    ASSERT_NO_THROW(ctx.infer_request.infer());

    ASSERT_NE(ctx.state->executing_thread_id.load(), caller_thread_id);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 33.0F);
}

namespace {

// Regression test for the P0 queue-poisoning bug: the old raw-lambda dispatch let exceptions
// escape into Defer()'s catch(...){}, corrupting the task queue. A second, successful infer() call
// after the first one failed is the tight assertion that would catch a regression.
void RunExceptionThenRecoveryScenario(const ov::AnyMap& properties) {
    auto ctx = CreateTestInferRequest(properties);
    const auto caller_thread_id = std::this_thread::get_id();

    ctx.state->should_throw = true;
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 1.0F);
    // The op's evaluate() raises std::runtime_error; the CPU plugin's node-execution wrapper
    // (Graph::ExecuteNodeWithCatch) re-throws non-Cancelled std::exceptions as ov::Exception.
    ASSERT_THROW(ctx.infer_request.infer(), ov::Exception);
    ASSERT_EQ(ctx.state->executing_thread_id.load(), caller_thread_id);

    ctx.state->should_throw = false;
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 123.0F);
    ASSERT_NO_THROW(ctx.infer_request.infer());

    ASSERT_EQ(ctx.state->executing_thread_id.load(), caller_thread_id);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 123.0F);
}

}  // namespace

TEST(AsyncInferRequestExceptionRecoveryTest, RecoversAfterExceptionForOptimizedSingleStream) {
    RunExceptionThenRecoveryScenario(OptimizedSingleStreamProperties());
}

TEST(AsyncInferRequestExceptionRecoveryTest, RecoversAfterExceptionForExplicitMultiAppThreadSync) {
    RunExceptionThenRecoveryScenario(ExplicitMultiAppThreadSyncProperties());
}

namespace {

// UAF/queue-poisoning regression test: two requests share one CompiledModel's executor, mirroring
// production executor sharing across compiled models. The old implementation could replay
// requestA's stale, exception-poisoned task after requestA's destruction when requestB infers next.
void RunDestroyThenReuseSameExecutorScenario(const ov::AnyMap& properties) {
    auto ctxA = CreateTestInferRequest(properties);
    ov::CompiledModel compiled_model = ctxA.compiled_model;  // keep the compiled model (and its shared executor) alive

    ctxA.state->should_throw = true;
    SetInput(ctxA.infer_request, compiled_model.input(), 1.0F);
    ASSERT_THROW(ctxA.infer_request.infer(), ov::Exception);

    // The Graph/ControllableOpState is shared infrastructure owned by the CompiledModel, not
    // requestA; reset should_throw before destroying requestA so requestB's infer() only exercises
    // the executor-queue behavior under test.
    ctxA.state->should_throw = false;

    // Destroy requestA while its shared executor stays alive via `compiled_model`.
    { ov::InferRequest dropped = std::move(ctxA.infer_request); }
    ctxA.state.reset();

    auto requestB = compiled_model.create_infer_request();
    SetInput(requestB, compiled_model.input(), 55.0F);
    ASSERT_NO_THROW(requestB.infer());
    ASSERT_FLOAT_EQ(GetOutput(requestB, compiled_model.output()), 55.0F);
}

}  // namespace

TEST(AsyncInferRequestExceptionRecoveryTest,
     DestroyingFailedRequestDoesNotPoisonSiblingRequestOnSameExecutorForOptimizedSingleStream) {
    RunDestroyThenReuseSameExecutorScenario(OptimizedSingleStreamProperties());
}

TEST(AsyncInferRequestExceptionRecoveryTest,
     DestroyingFailedRequestDoesNotPoisonSiblingRequestOnSameExecutorForExplicitMultiAppThreadSync) {
    RunDestroyThenReuseSameExecutorScenario(ExplicitMultiAppThreadSyncProperties());
}

TEST(AsyncInferRequestRaceConditionTest, ObservesBusyStateWhileCallerThreadSyncInferIsInFlight) {
    auto ctx = CreateTestInferRequest(ExplicitMultiAppThreadSyncProperties());
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 42.0F);
    ctx.state->arm_gate();

    // Guarantees the blocked op is released even if an assertion below fails early --
    // otherwise the in-flight future's destructor would block forever.
    struct ReleaseGuard {
        std::shared_ptr<ControllableOpState> state;
        ~ReleaseGuard() {
            state->release();
        }
    } release_guard{ctx.state};

    auto in_flight = std::async(std::launch::async, [&] {
        ctx.infer_request.infer();
    });

    // Wait until the in-flight infer() has genuinely entered the (blocked) computation, so the
    // request is truly busy, not just "about to start".
    ctx.state->wait_until_entered();

    // Exact exception type (ov::Busy) isn't checked: on some platforms (e.g. Android, where the
    // CPU plugin loads as a separate shared library) RTTI identity doesn't cross that boundary,
    // so it arrives as a plain ov::Exception with the same message.
    OV_EXPECT_THROW_HAS_SUBSTRING(ctx.infer_request.infer(), ov::Exception, "busy");

    ctx.state->release();
    ASSERT_NO_THROW(in_flight.get());
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 42.0F);
}

TEST(AsyncInferRequestCancelTest, CancelDuringCallerThreadSyncInferRaisesCancelled) {
    auto ctx = CreateTestInferRequest(ExplicitMultiAppThreadSyncProperties());
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 7.0F);
    ctx.state->arm_gate();

    auto in_flight = std::async(std::launch::async, [&] {
        ctx.infer_request.infer();
    });

    ctx.state->wait_until_entered();
    ctx.infer_request.cancel();
    ctx.state->release();

    // See the busy-exception comment above: checked by message, not exact type.
    OV_EXPECT_THROW_HAS_SUBSTRING(in_flight.get(), ov::Exception, "cancel");

    // cancel() only affects the in-flight call; the request must be fully usable afterwards.
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 9.0F);
    ASSERT_NO_THROW(ctx.infer_request.infer());
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 9.0F);
}

// Known limitation: more app threads than configured streams can collide on the same graph slot
// and serialize. This test locks in that current behavior -- only checks numerical correctness,
// not per-thread isolation.
TEST(AsyncInferRequestStreamCollisionTest, StaysCorrectWhenAppThreadsExceedConfiguredStreams) {
    constexpr int kStreams = 4;
    constexpr int kAppThreads = 8;  // M > N: guarantees stream-id collisions (8 % 4 == 0)

    auto state = std::make_shared<ControllableOpState>();  // shared op state: no blocking needed here
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto op = std::make_shared<ControllableOp>(ov::OutputVector{param}, state);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "AsyncInferRequestCollisionTestModel");

    auto compiled_model = GetTestCore().compile_model(model,
                                                      kTestDeviceName,
                                                      {ov::num_streams(kStreams),
                                                       ov::hint::enable_cpu_pinning(false),
                                                       ov::intel_cpu::multi_app_thread_sync_execution(true)});

    std::vector<ov::InferRequest> requests;
    requests.reserve(kAppThreads);
    for (int i = 0; i < kAppThreads; ++i) {
        requests.push_back(compiled_model.create_infer_request());
    }

    std::vector<std::thread> workers;
    std::vector<bool> succeeded(kAppThreads, false);
    std::vector<float> observed(kAppThreads, -1.0F);
    workers.reserve(kAppThreads);
    for (int i = 0; i < kAppThreads; ++i) {
        workers.emplace_back([&, i] {
            const float value = static_cast<float>(i + 1);
            SetInput(requests[i], compiled_model.input(), value);
            try {
                requests[i].infer();
                observed[i] = GetOutput(requests[i], compiled_model.output());
                succeeded[i] = true;
            } catch (...) {
                succeeded[i] = false;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    for (int i = 0; i < kAppThreads; ++i) {
        EXPECT_TRUE(succeeded[i]) << "application thread " << i << " did not complete successfully";
        EXPECT_FLOAT_EQ(observed[i], static_cast<float>(i + 1)) << "application thread " << i;
    }
}
