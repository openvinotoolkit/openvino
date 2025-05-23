// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>
#include <common_test_utils/test_constants.hpp>
#include <openvino/core/model.hpp>
#include <openvino/op/op.hpp>
#include <openvino/openvino.hpp>

#include <thread>
#include <condition_variable>

namespace ov {
namespace test {
// Openvino extension operation that sleeps for X us in its evaluate method
namespace {
enum class TestSteps { INIT, ENTER_EVALUATE, RUN_EVALUATE };
}  // namespace

class SleepCustomOp : public ov::op::Op {
public:
    OPENVINO_OP("SleepCustomOp");
    SleepCustomOp() = default;
    SleepCustomOp(const ov::OutputVector& args,
          size_t sleep,
          std::shared_ptr<std::mutex> mutex,
          std::shared_ptr<std::condition_variable> cv,
          std::shared_ptr<std::atomic<TestSteps>> test_step)
        : Op(args),
          m_sleep(sleep),
          m_mutex(mutex),
          m_cv(cv),
          m_test_step(test_step) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
        auto new_op = std::make_shared<SleepCustomOp>(new_args, m_sleep, m_mutex, m_cv, m_test_step);
        return new_op;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    void revalidate_and_infer_types() override {
        validate_and_infer_types();
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        // signal entering the evaluate method
        {
            std::lock_guard<std::mutex> lock(*m_mutex);
            m_test_step->store(TestSteps::ENTER_EVALUATE);
        }
        m_cv->notify_all();
        {
            // this is required to start all the evaluate calls at the same time
            std::unique_lock<std::mutex> lock(*m_mutex);
            m_cv->wait(lock, [&] {
                return m_test_step->load() == TestSteps::RUN_EVALUATE;
            });
        }
        std::this_thread::sleep_for(std::chrono::microseconds(m_sleep));
        return true;
    }

    bool evaluate(ov::TensorVector& output_values,
                  const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override {
        return evaluate(output_values, input_values);
    }

    bool has_evaluate() const override {
        return true;
    }

private:
    size_t m_sleep;  // sleep time in us
    std::shared_ptr<std::mutex> m_mutex;
    std::shared_ptr<std::condition_variable> m_cv;
    std::shared_ptr<std::atomic<TestSteps>> m_test_step;
};

class ReleaseMemoryMultiThreadTest : public ::testing::Test {
protected:
    void SetUp() override {
        param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});

        constexpr size_t sleep_time = 5;  // us
        mutex = std::make_shared<std::mutex>();
        cv = std::make_shared<std::condition_variable>();
        test_step = std::make_shared<std::atomic<TestSteps>>(TestSteps::INIT);

        auto sleep = std::make_shared<SleepCustomOp>(ov::OutputVector{param}, sleep_time, mutex, cv, test_step);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sleep)};
        ov::ParameterVector params{param};

        auto model = std::make_shared<ov::Model>(results, params, "testModel");

        compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU, {{"NUM_STREAMS", num_streams}});
    }

protected:
    const size_t num_streams = 1; // use only one async stream to simplify invocation order syncronization
    ov::Core core;
    ov::CompiledModel compiled_model;
    std::shared_ptr<ov::op::v0::Parameter> param;

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<std::condition_variable> cv;
    std::shared_ptr<std::atomic<TestSteps>> test_step;
};
}  // namespace test
}  // namespace ov

using namespace ov::test;

TEST_F(ReleaseMemoryMultiThreadTest, smoke_throwInferenceIsRunning) {
    // Create and infer a few infer requests concurrently
    std::vector<ov::InferRequest> inferRequests;
    for (size_t i = 0; i < num_streams; i++) {
        auto inferRequest = compiled_model.create_infer_request();
        inferRequest.set_tensor(param, ov::Tensor(ov::element::f32, ov::Shape{1}));
        inferRequests.push_back(std::move(inferRequest));
    }
    // infer the infer requests
    for (auto& inferRequest : inferRequests) {
        inferRequest.start_async();
    }

    //wait till the infer request enters evaluate
    {
        std::unique_lock<std::mutex> lock(*mutex);
        cv->wait(lock, [&] {
            return test_step->load() == TestSteps::ENTER_EVALUATE;
        });
    }

    // While the infer requests are waiting on the cv, call release_memory.
    // We expect that the method will throw an exception when it is called while infer requests are running.
    EXPECT_THROW(compiled_model.release_memory(), ov::Exception);

    // lets unlock cv
    {
        std::lock_guard<std::mutex> lock(*mutex);
        test_step->store(TestSteps::RUN_EVALUATE);
    }
    cv->notify_all();

    for (auto& inferRequest : inferRequests) {
        inferRequest.wait();
    }
}

TEST_F(ReleaseMemoryMultiThreadTest, smoke_noThrowInferenceIsNotRunning) {
    // Create and infer a few infer requests concurrently
    std::vector<ov::InferRequest> inferRequests;
    for (size_t i = 0; i < num_streams; i++) {
        auto inferRequest = compiled_model.create_infer_request();
        inferRequest.set_tensor(param, ov::Tensor(ov::element::f32, ov::Shape{1}));
        inferRequests.push_back(std::move(inferRequest));
    }
    // infer the infer requests
    for (auto& inferRequest : inferRequests) {
        inferRequest.start_async();
    }

    //wait till the infer request enters evaluate
    {
        std::unique_lock<std::mutex> lock(*mutex);
        cv->wait(lock, [&] {
            return test_step->load() == TestSteps::ENTER_EVALUATE;
        });
    }

    // lets unlock cv
    {
        std::lock_guard<std::mutex> lock(*mutex);
        test_step->store(TestSteps::RUN_EVALUATE);
    }
    cv->notify_all();

    for (auto& inferRequest : inferRequests) {
        inferRequest.wait();
    }

    // Don't throw when the infer requests are finished
    EXPECT_NO_THROW(compiled_model.release_memory());
}