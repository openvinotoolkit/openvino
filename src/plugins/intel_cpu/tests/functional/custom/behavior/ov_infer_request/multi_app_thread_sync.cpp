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
#include <string>
#include <thread>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

// Builds a real model through the standard "CPU" device instead of mocks, so CPUStreamsExecutor's
// actual task-queue/exception mechanics are genuinely exercised.
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
    // parallel_get_max_threads() as observed from inside evaluate(), i.e. the intra-op thread budget
    // node execution actually ran with (OpenMP ICV or TBB arena concurrency, depending on the backend).
    std::atomic<int> max_threads_during_evaluate{-1};

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
        m_state->max_threads_during_evaluate = parallel_get_max_threads();
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

struct TestInferContext {
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::shared_ptr<ControllableOpState> state;
};

// Builds Parameter -> ControllableOp -> Result, compiles it with the given properties through the
// CPU plugin, and creates one real AsyncInferRequest on top of it.
TestInferContext CreateTestInferRequest(const ov::AnyMap& properties) {
    auto state = std::make_shared<ControllableOpState>();
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto op = std::make_shared<ControllableOp>(ov::OutputVector{param}, state);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "AsyncInferRequestTestModel");

    ov::Core core;
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU, properties);
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

// 4 streams with multi_app_thread_sync_execution=true (pinning off, since the plugin rejects that
// combination with pinning resolved to true).
ov::AnyMap ExplicitMultiAppThreadSyncProperties() {
    return {ov::num_streams(4),
            ov::inference_num_threads(4),
            ov::hint::enable_cpu_pinning(false),
            ov::intel_cpu::multi_app_thread_sync_execution(true)};
}

}  // namespace

TEST(AsyncInferRequestTest, smoke_UsesCallerThreadForExplicitMultiAppThreadSync) {
    auto ctx = CreateTestInferRequest(ExplicitMultiAppThreadSyncProperties());
    ASSERT_TRUE(ctx.compiled_model.get_property(ov::intel_cpu::multi_app_thread_sync_execution.name()).as<bool>());

    SetInput(ctx.infer_request, ctx.compiled_model.input(), 22.0F);
    const auto caller_thread_id = std::this_thread::get_id();
    ASSERT_NO_THROW(ctx.infer_request.infer());

    ASSERT_EQ(ctx.state->executing_thread_id.load(), caller_thread_id);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 22.0F);
}

namespace {

// multi_app_thread_sync_execution with num_streams deliberately smaller than inference_num_threads, so that
// without the force-to-1 in generate_stream_info() each stream would get threads / streams (floored) threads.
ov::AnyMap MultiAppThreadSyncProperties(int streams, int threads) {
    return {ov::num_streams(streams),
            ov::inference_num_threads(threads),
            ov::hint::enable_cpu_pinning(false),
            ov::intel_cpu::multi_app_thread_sync_execution(true)};
}

// Runs infer() twice on one genuinely fresh application thread (never seen by the model's executor): the 1st
// call hits CPUStreamsExecutor's lazy per-thread Stream construction, the 2nd one does not. The intra-op thread
// budget seen inside node execution must be exactly 1 for both, and results must stay correct.
void RunFreshThreadForcedSingleThreadScenario(int streams, int threads) {
    auto ctx = CreateTestInferRequest(MultiAppThreadSyncProperties(streams, threads));

    // The executor config itself resolves to 1 thread per stream, whatever was requested.
    ASSERT_EQ(ctx.compiled_model.get_property(ov::num_streams), streams);
    ASSERT_EQ(ctx.compiled_model.get_property(ov::inference_num_threads), streams);

    constexpr int kCalls = 2;
    std::vector<int> observed_threads;
    std::vector<float> observed_outputs;
    std::vector<std::thread::id> executing_threads;
    std::string failure;
    std::thread::id fresh_thread_id;

    std::thread fresh_thread([&] {
        fresh_thread_id = std::this_thread::get_id();
        try {
            for (int call = 0; call < kCalls; ++call) {
                ctx.state->max_threads_during_evaluate = -1;
                SetInput(ctx.infer_request, ctx.compiled_model.input(), static_cast<float>(call + 1));
                ctx.infer_request.infer();
                observed_threads.push_back(ctx.state->max_threads_during_evaluate.load());
                executing_threads.push_back(ctx.state->executing_thread_id.load());
                observed_outputs.push_back(GetOutput(ctx.infer_request, ctx.compiled_model.output()));
            }
        } catch (const std::exception& ex) {
            failure = ex.what();
        }
    });
    fresh_thread.join();

    ASSERT_TRUE(failure.empty()) << failure;
    ASSERT_EQ(observed_threads.size(), static_cast<size_t>(kCalls));
    for (int call = 0; call < kCalls; ++call) {
        EXPECT_EQ(observed_threads[call], 1) << "call #" << call + 1 << " on the fresh thread (requested " << streams
                                             << " streams / " << threads << " threads)";
        EXPECT_EQ(executing_threads[call], fresh_thread_id) << "call #" << call + 1;
        EXPECT_FLOAT_EQ(observed_outputs[call], static_cast<float>(call + 1)) << "call #" << call + 1;
    }
}

}  // namespace

// Regression for the OpenMP-oversubscription review finding, fixed at the config level instead of in the
// caller-thread guard: the per-stream thread budget must be exactly 1 for every call ordinal on a thread
// (including the very first one, where Stream's lazy ctor runs), and independent of how requested threads
// divide among streams.
TEST(AsyncInferRequestTest, smoke_MultiAppThreadSyncForcesSingleThreadWhenThreadsDivideEvenlyAmongStreams) {
    RunFreshThreadForcedSingleThreadScenario(2, 8);  // would be 4 threads/stream without the force
}

TEST(AsyncInferRequestTest, smoke_MultiAppThreadSyncForcesSingleThreadWhenThreadsDoNotDivideEvenlyAmongStreams) {
    RunFreshThreadForcedSingleThreadScenario(2, 7);  // would floor to 3 threads/stream (6 of 7 threads used)
}

// The force must be scoped to multi_app_thread_sync_execution: the same streams/threads configuration without
// it keeps its real, multi-threaded per-stream budget.
TEST(AsyncInferRequestTest, smoke_RegularPathKeepsMultipleThreadsPerStream) {
    constexpr int kStreams = 2;
    constexpr int kThreads = 8;
#if OV_THREAD == OV_THREAD_SEQ
    GTEST_SKIP() << "SEQ threading never executes on worker threads, so it cannot resolve multiple threads per "
                    "stream regardless of configuration";
#endif
    auto ctx = CreateTestInferRequest(
        {ov::num_streams(kStreams), ov::inference_num_threads(kThreads), ov::hint::enable_cpu_pinning(false)});
    ASSERT_FALSE(ctx.compiled_model.get_property(ov::intel_cpu::multi_app_thread_sync_execution.name()).as<bool>());
    // The plugin caps the request to the cores it can use (e.g. physical cores only), so check against the
    // resolved budget rather than the requested one.
    const int resolved_threads = ctx.compiled_model.get_property(ov::inference_num_threads);
    if (resolved_threads < 2 * kStreams || resolved_threads % kStreams != 0) {
        GTEST_SKIP() << "plugin resolved " << resolved_threads << " threads; needs a multiple of " << kStreams
                     << " that gives at least 2 threads per stream";
    }

    SetInput(ctx.infer_request, ctx.compiled_model.input(), 5.0F);
    ASSERT_NO_THROW(ctx.infer_request.infer());
    ASSERT_NE(ctx.state->executing_thread_id.load(), std::this_thread::get_id());
    ASSERT_EQ(ctx.state->max_threads_during_evaluate.load(), resolved_threads / kStreams);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 5.0F);
}

#if OV_THREAD == OV_THREAD_OMP
// OpenMP-only: CallerThreadStateGuard must (a) force the active OMP thread count to 1 for every single
// infer() call on the caller thread -- not just the first, where Stream's lazy omp_set_num_threads() would
// have happened to do it anyway -- and (b) put the caller thread's own omp_get_max_threads() back afterwards.
// (Not built in TBB configurations, where there is no per-thread OpenMP state to force or restore.)
TEST(AsyncInferRequestOpenMPStateTest, smoke_RestoresCallerOmpThreadsAfterCallerThreadSyncCalls) {
    constexpr int kAppThreadOmpThreads = 3;
    auto ctx = CreateTestInferRequest(MultiAppThreadSyncProperties(2, 8));

    int first_call_threads = -1;
    int second_call_threads = -1;
    int omp_after_first = -1;
    int omp_after_second = -1;
    std::string failure;
    std::thread fresh_thread([&] {
        try {
            omp_set_num_threads(kAppThreadOmpThreads);
            SetInput(ctx.infer_request, ctx.compiled_model.input(), 1.0F);
            ctx.infer_request.infer();
            first_call_threads = ctx.state->max_threads_during_evaluate.load();
            omp_after_first = omp_get_max_threads();

            // Stream's lazy omp_set_num_threads() (cpu_streams_executor.cpp) only runs once per thread,
            // on the first execute() call -- it does not refire here. Without the guard reapplying
            // omp_set_num_threads(1) on every entry, this second call would run with omp_after_first
            // (the restored, pre-infer caller value) instead of 1.
            ctx.state->max_threads_during_evaluate = -1;
            ctx.infer_request.infer();
            second_call_threads = ctx.state->max_threads_during_evaluate.load();
            omp_after_second = omp_get_max_threads();
        } catch (const std::exception& ex) {
            failure = ex.what();
        }
    });
    fresh_thread.join();

    ASSERT_TRUE(failure.empty()) << failure;
    EXPECT_EQ(first_call_threads, 1);
    EXPECT_EQ(second_call_threads, 1);
    EXPECT_EQ(omp_after_first, kAppThreadOmpThreads);
    EXPECT_EQ(omp_after_second, kAppThreadOmpThreads);
}
#endif

TEST(AsyncInferRequestTest, smoke_KeepsExecutorDispatchForSyncInferWhenCallerThreadModeIsDisabled) {
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

TEST(AsyncInferRequestExceptionRecoveryTest, smoke_RecoversAfterExceptionForExplicitMultiAppThreadSync) {
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
     smoke_DestroyingFailedRequestDoesNotPoisonSiblingRequestOnSameExecutorForExplicitMultiAppThreadSync) {
    RunDestroyThenReuseSameExecutorScenario(ExplicitMultiAppThreadSyncProperties());
}

TEST(AsyncInferRequestRaceConditionTest, smoke_ObservesBusyStateWhileCallerThreadSyncInferIsInFlight) {
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

    ASSERT_THROW(ctx.infer_request.infer(), ov::Busy);

    ctx.state->release();
    ASSERT_NO_THROW(in_flight.get());
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 42.0F);
}

TEST(AsyncInferRequestCancelTest, smoke_CancelDuringCallerThreadSyncInferRaisesCancelled) {
    auto ctx = CreateTestInferRequest(ExplicitMultiAppThreadSyncProperties());
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 7.0F);
    ctx.state->arm_gate();

    auto in_flight = std::async(std::launch::async, [&] {
        ctx.infer_request.infer();
    });

    ctx.state->wait_until_entered();
    ctx.infer_request.cancel();
    ctx.state->release();

    ASSERT_THROW(in_flight.get(), ov::Cancelled);

    // cancel() only affects the in-flight call; the request must be fully usable afterwards.
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 9.0F);
    ASSERT_NO_THROW(ctx.infer_request.infer());
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 9.0F);
}

// With more app threads than configured streams (M > N), get_graph() finds every graph slot busy and
// some threads block on a slot until it frees up. This test only checks numerical correctness
// under that oversubscription, not per-thread isolation.
TEST(AsyncInferRequestStreamCollisionTest, smoke_StaysCorrectWhenAppThreadsExceedConfiguredStreams) {
    constexpr int kStreams = 4;
    constexpr int kAppThreads = 8;  // M > N: guarantees stream-id collisions (8 % 4 == 0)

    auto state = std::make_shared<ControllableOpState>();  // shared op state: no blocking needed here
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto op = std::make_shared<ControllableOp>(ov::OutputVector{param}, state);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "AsyncInferRequestCollisionTestModel");

    ov::Core core;
    auto compiled_model = core.compile_model(model,
                                             ov::test::utils::DEVICE_CPU,
                                             {ov::num_streams(kStreams),
                                              ov::hint::enable_cpu_pinning(false),
                                              ov::intel_cpu::multi_app_thread_sync_execution(true)});

    std::vector<ov::InferRequest> requests;
    requests.reserve(kAppThreads);
    for (int i = 0; i < kAppThreads; ++i) {
        requests.push_back(compiled_model.create_infer_request());
    }

    std::vector<std::thread> workers;
    // Not std::vector<bool>: its bit-packed elements share words, so concurrent writes from the workers race.
    std::vector<char> succeeded(kAppThreads, 0);
    std::vector<float> observed(kAppThreads, -1.0F);
    workers.reserve(kAppThreads);
    for (int i = 0; i < kAppThreads; ++i) {
        workers.emplace_back([&, i] {
            const float value = static_cast<float>(i + 1);
            SetInput(requests[i], compiled_model.input(), value);
            try {
                requests[i].infer();
                observed[i] = GetOutput(requests[i], compiled_model.output());
                succeeded[i] = 1;
            } catch (...) {
                succeeded[i] = 0;
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
