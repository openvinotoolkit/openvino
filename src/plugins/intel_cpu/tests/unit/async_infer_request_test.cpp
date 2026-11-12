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

#include "common_test_utils/file_utils.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/util/file_util.hpp"

// NOTE on test design (see the review-fix plan for PR #35276, P0 item):
//
// The previous version of this file used a hand-rolled `TestStreamsExecutor` mock whose
// `execute()` just called `task()` directly, and a `TestInferRequest` mock that was a plain
// `ov::IInferRequest`, not a `SyncInferRequest`. Neither reproduced the real bug: a mock
// `execute()` doesn't replicate `CPUStreamsExecutor::Impl::Defer()`'s real
// `catch (...) {}`-then-maybe-not-`pop()` task-queue mechanics, so a test built on it could not
// tell a correct fix apart from the original P0 bug (both "pass"). And `static_cast<SyncInferRequest*>`
// (restored in production per reviewer wangleis's request) is undefined behavior on a non-
// `SyncInferRequest` object.
//
// This rewrite instead builds a tiny *real* model (Parameter -> ControllableOp -> Result) and
// compiles it through a real `ov::intel_cpu::Plugin`.
//
// Two approaches were tried for the compile step:
//  1. Constructing `ov::intel_cpu::Plugin` directly (no `ov::Core`) plus a bare `ov::MockICore`.
//     This links cleanly for everything EXCEPT `Plugin` itself: the CPU plugin's build uses
//     `ov_add_version_defines()` to compile src/plugin.cpp into a *separate* companion object
//     library (`openvino_intel_cpu_plugin_obj_version`) that is merged into the real
//     `openvino_intel_cpu_plugin` shared library via a `target_sources(... $<TARGET_OBJECTS:...>)`
//     generator expression. `tests/unit/CMakeLists.txt` only lists
//     `$<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>` as this test binary's extra object files,
//     which does not transitively pick up that companion object library, so `Plugin::Plugin()` /
//     `Plugin::compile_model()` are undefined references when linking directly against the object
//     library from this test binary. Fixing that is a build-system (CMakeLists.txt) change outside
//     this test-only task's scope, so this approach was dropped rather than touching production
//     build files.
//  2. Using the already-built real `libopenvino_intel_cpu_plugin.so` through `ov::Core`
//     (`core.register_plugin(path, device_name)`), which is exactly the documented fallback for
//     this test file. This is what is implemented below: it still produces a real
//     `ov::intel_cpu::CompiledModel` -> real `ov::intel_cpu::SyncInferRequest` -> real
//     `ov::intel_cpu::AsyncInferRequest` stack sitting on top of a real
//     `ov::threading::CPUStreamsExecutor` (so `Defer()`/`Execute()`'s real exception/task-queue
//     mechanics and the `static_cast<SyncInferRequest*>` in `AsyncInferRequest`'s constructor are
//     genuinely exercised, not mocked), it still lives in tests/unit/ and is registered the normal
//     way in sources.cmake, and it still compiles/runs in well under a second per test -- it is
//     just reached via the public `ov::Core`/`ov::CompiledModel`/`ov::InferRequest` API surface
//     instead of the plugin-internal classes directly.
//
// `ControllableOp` (modeled after `SleepCustomOp` in
// tests/functional/custom/behavior/ov_executable_network/concurent_release_memory.cpp, which
// already proves this "custom op compiled through the real CPU plugin" approach works end to
// end) is picked up by the CPU plugin's generic `Reference` node fallback (any op with
// `has_evaluate() == true` that isn't natively supported gets wrapped by
// `ov::intel_cpu::node::Reference`, see node.cpp/reference.cpp), so no CPU-plugin-internal
// registration is required. It lets each test deterministically: observe which thread actually
// ran the computation, block until a controlling thread releases it (for busy/cancel races),
// and optionally fail (for the exception/queue-poisoning regression tests) -- all without
// relying on fragile wall-clock timing.

// ControllableOp (and its state struct) are deliberately NOT inside the anonymous namespace below:
// OPENVINO_OP()'s RTTI machinery applies a hidden-visibility attribute to a static member function,
// which some compilers (e.g. the RISC-V GCC toolchain used in CI) warn/error on ("visibility
// attribute ignored") when the enclosing class already has no linkage because it lives in an
// anonymous namespace. Giving this class ordinary (translation-unit-local, since this is a .cpp
// file) external linkage avoids that false positive; its RTTI type name is unique enough that this
// carries no practical risk of clashing with anything else in the test binary.

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

// Device name under which the real CPU plugin .so is registered with a dedicated, test-local
// ov::Core instance (see GetTestCore() below) -- deliberately distinct from "CPU" so this never
// collides with any ambient/default plugin registration.
constexpr const char* kTestDeviceName = "ASYNC_INFER_REQUEST_TEST_CPU";

// Lazily builds (once) a process-local ov::Core with the real, just-built CPU plugin shared
// library registered under kTestDeviceName. Reused across all TESTs in this file: compile_model()
// itself is what creates a brand-new, independent CompiledModel/executor per test, so sharing the
// Core/plugin instance is safe and keeps each test fast.
ov::Core& GetTestCore() {
    static ov::Core core = [] {
        ov::Core c;
        auto plugin_path =
            ov::util::make_plugin_library_name(ov::util::make_path(ov::test::utils::getExecutableDirectory()),
                                               std::string("openvino_intel_cpu_plugin") + OV_BUILD_POSTFIX);
        c.register_plugin(plugin_path.string(), kTestDeviceName);
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

// Properties that yield is_optimized_single_stream == true (1 stream, 1 thread total) without
// requesting CPU_MULTI_APP_THREAD_SYNC_EXECUTION at all.
ov::AnyMap OptimizedSingleStreamProperties() {
    return {ov::num_streams(1), ov::inference_num_threads(1)};
}

// Properties that yield run_sync_in_caller_thread == true via an explicit multi-stream request
// for CPU_MULTI_APP_THREAD_SYNC_EXECUTION. CPU pinning is explicitly disabled: the plugin
// rejects the combination of multi_app_thread_sync_execution=true with CPU pinning resolved to
// true (see compiled_model.cpp), and pinning otherwise defaults to true on typical Linux
// topologies, which is not what these tests are exercising.
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
    // Same stream count as ExplicitMultiAppThreadSyncProperties, but CPU_MULTI_APP_THREAD_SYNC_EXECUTION
    // is not requested, so this must keep using the base class's default ImmediateStreamsExecutor,
    // which dispatches to a worker thread (via run_and_wait) whenever streams > 1.
    auto ctx = CreateTestInferRequest({ov::num_streams(4), ov::inference_num_threads(4)});
    ASSERT_FALSE(ctx.compiled_model.get_property(ov::intel_cpu::multi_app_thread_sync_execution.name()).as<bool>());

    SetInput(ctx.infer_request, ctx.compiled_model.input(), 33.0F);
    const auto caller_thread_id = std::this_thread::get_id();
    ASSERT_NO_THROW(ctx.infer_request.infer());

    ASSERT_NE(ctx.state->executing_thread_id.load(), caller_thread_id);
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 33.0F);
}

namespace {

// This is the actual regression test for the P0 queue-poisoning bug. With the old `m_infer_func`
// override that called `m_stream_executor->execute(...)` with a raw, unwrapped lambda, an
// exception thrown from inference would escape straight into
// `CPUStreamsExecutor::Impl::Defer()`'s outer `catch (...) {}`, which swallows it *before*
// popping the failed task off the per-thread `_taskQueue` -- corrupting that queue for the next
// call on the same thread (stale/poisoned task re-run, risking UAF). With the current
// `m_sync_pipeline` + `CallerThreadStageExecutor` design, `make_next_stage_task()` already wraps
// every stage in its own try/catch before any executor ever sees it, so nothing ever throws out
// of the task object into `Defer()`, and the queue is never corrupted. A second, successful
// `infer()` call on the very same request/executor after the first one failed is therefore the
// tight assertion that would catch a regression back to the old behavior (which would otherwise
// re-run the stale task, hang, or crash on the second call).
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

// This is the UAF/queue-poisoning regression test explicitly requested by reviewer wangleis for the P0 test-validity
// comment: "destroying the failed request then using another on same executor is safe". Both requests are created
// from the same ov::CompiledModel, so both of their AsyncInferRequest/SyncInferRequest stacks share the exact same
// underlying ov::threading::CPUStreamsExecutor instance (CompiledModel::m_task_executor is one shared member -- this
// mirrors the real production risk called out in the review: "executors are shared across compiled models" via
// ExecutorManager::get_idle_cpu_streams_executor(), which would return the very same executor for two separately
// compiled models using identical stream configs, as long as the first one is still idle; using two requests on one
// compiled model here is just a simpler, deterministic way to force that sharing for the test).
//
// Under the old buggy m_infer_func implementation, requestA's failing task would be left at the front of the
// executor's per-thread task queue (Defer()'s catch(...){} swallows the exception before pop()), and the lambda that
// failed captured `this` (requestA's AsyncInferRequest). Destroying requestA then frees that object; the next task
// run on the same thread through the same executor (requestB's infer()) would replay the poisoned, stale task first
// -- a use-after-free. With the current make_next_stage_task()-based pipeline, the stage task always finishes
// normally from the executor's point of view (exceptions are caught inside the task itself), so the queue is never
// left poisoned and requestA's destruction cannot affect requestB.
void RunDestroyThenReuseSameExecutorScenario(const ov::AnyMap& properties) {
    auto ctxA = CreateTestInferRequest(properties);
    ov::CompiledModel compiled_model = ctxA.compiled_model;  // keep the compiled model (and its shared executor) alive

    ctxA.state->should_throw = true;
    SetInput(ctxA.infer_request, compiled_model.input(), 1.0F);
    ASSERT_THROW(ctxA.infer_request.infer(), ov::Exception);

    // The per-stream Graph (and the ControllableOp node instance inside it, hence ControllableOpState) is owned by
    // the CompiledModel and reused across every infer request created from it -- unlike the executor's task queue,
    // it is deliberately shared infrastructure, not something that disappears when requestA is destroyed. Reset it
    // back to "don't throw" before destroying requestA so requestB's infer() exercises only the thing actually under
    // test here (whether requestA's destruction left the *executor's* queue in a bad state), not a leftover
    // should_throw flag from requestA's own setup.
    ctxA.state->should_throw = false;

    // Destroy requestA (and its ControllableOpState handle) while its shared executor stays alive via `compiled_model`.
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

    ASSERT_THROW(in_flight.get(), ov::Cancelled);

    // cancel() only affects the in-flight call; the request must be fully usable afterwards.
    SetInput(ctx.infer_request, ctx.compiled_model.input(), 9.0F);
    ASSERT_NO_THROW(ctx.infer_request.infer());
    ASSERT_FLOAT_EQ(GetOutput(ctx.infer_request, ctx.compiled_model.output()), 9.0F);
}

// Known/accepted limitation (PR #35276 review item F-4, documented in
// CompiledModel::get_graph(), not fixed here per explicit review discussion): the per-thread
// stream id handed out by CPUStreamsExecutor grows from an unbounded counter, so when more
// concurrent application threads call infer() than there are configured streams/graphs, two or
// more threads can be routed to the very same graph slot and serialize on its lock instead of
// running on independent ones. This test documents and locks in that CURRENT behavior: it does
// NOT assert anything about per-thread isolation/parallelism, only that results stay numerically
// correct and no thread crashes or throws when 8 application threads (M) share just 4 streams (N).
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
