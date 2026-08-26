// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> createMaxPoolModelMT(bool dynamicBatch = false, bool nhwcLayout = true) {
    std::shared_ptr<ov::op::v0::Parameter> input;
    if (dynamicBatch) {
        input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                        ov::PartialShape{ov::Dimension(1, 10), 16, 1280, 1280});
    } else {
        input = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f16,
            ov::PartialShape{1, 16, ov::Dimension(10, 1280), ov::Dimension(10, 1280)});
    }

    std::string inputName = "input1";
    input->set_friendly_name(inputName);
    input->get_output_tensor(0).set_names({inputName});
    if (!nhwcLayout)
        input->set_layout("NCHW");
    auto maxpool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                         Strides{1, 1},
                                                         Shape{0, 0},
                                                         Shape{0, 0},
                                                         Shape{1, 1},
                                                         op::RoundingType::FLOOR,
                                                         op::PadType::EXPLICIT);
    maxpool->set_friendly_name("MaxPool_2");

    auto result = std::make_shared<ov::op::v0::Result>(maxpool);
    std::string outputName = "output";
    if (!nhwcLayout)
        result->set_layout("NCHW");
    result->set_friendly_name(outputName);
    result->get_output_tensor(0).set_names({outputName});

    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "MaxPool");

    if (nhwcLayout) {
        auto preProc = ov::preprocess::PrePostProcessor(model);
        preProc.input(0).tensor().set_layout("NHWC");
        preProc.input(0).model().set_layout("NCHW");
        preProc.output(0).tensor().set_layout("NHWC");
        preProc.output(0).model().set_layout("NCHW");
        model = preProc.build();
    }

    return model;
}

inline std::shared_ptr<ov::Model> createCustomNetModelMT() {
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16,
        ov::PartialShape{1, 16, ov::Dimension(1, 1280), ov::Dimension(10, 1920)});
    input->set_friendly_name("Parameter_59");

    auto make_conv_add = [](const ov::Output<ov::Node>& data,
                            const std::string& convName,
                            const std::string& addName,
                            float weightValue,
                            float biasValue) -> ov::Output<ov::Node> {
        const std::vector<float> weightValues(16 * 16, weightValue);
        const std::vector<float> biasValues(16, biasValue);

        auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{16, 16, 1, 1}, weightValues);
        auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                              weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        conv->set_friendly_name(convName);

        auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, biasValues);
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
        add->set_friendly_name(addName);
        return add;
    };

    auto x = make_conv_add(input, "Convolution_61", "Add_63", 0.01f, 0.001f);
    x = make_conv_add(x, "Convolution_65", "Add_67", 0.011f, 0.001f);

    auto relu68 = std::make_shared<ov::op::v0::Relu>(x);
    relu68->set_friendly_name("Relu_68");
    x = relu68;

    x = make_conv_add(x, "Convolution_70", "Add_72", 0.012f, 0.001f);
    auto relu73 = std::make_shared<ov::op::v0::Relu>(x);
    relu73->set_friendly_name("Relu_73");
    x = relu73;

    x = make_conv_add(x, "Convolution_75", "Add_77", 0.013f, 0.001f);
    auto relu78 = std::make_shared<ov::op::v0::Relu>(x);
    relu78->set_friendly_name("Relu_78");
    x = relu78;

    x = make_conv_add(x, "Convolution_82", "Add_84", 0.014f, 0.001f);
    auto relu85 = std::make_shared<ov::op::v0::Relu>(x);
    relu85->set_friendly_name("Relu_85");
    x = relu85;

    x = make_conv_add(x, "Convolution_87", "Add_89", 0.015f, 0.001f);
    auto relu90 = std::make_shared<ov::op::v0::Relu>(x);
    relu90->set_friendly_name("Relu_90");
    x = relu90;

    x = make_conv_add(x, "Convolution_92", "Add_94", 0.016f, 0.001f);
    auto relu95 = std::make_shared<ov::op::v0::Relu>(x);
    relu95->set_friendly_name("Relu_95");
    x = relu95;

    auto multiplyScale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, {0.5f});
    auto multiply97 = std::make_shared<ov::op::v1::Multiply>(x, multiplyScale);
    multiply97->set_friendly_name("Multiply_97");

    auto add98 = std::make_shared<ov::op::v1::Add>(multiply97, multiply97);
    add98->set_friendly_name("Add_98");

    x = make_conv_add(add98, "Convolution_100", "Add_102", 0.017f, 0.001f);

    auto result = std::make_shared<ov::op::v0::Result>(x);
    result->set_friendly_name("Result_104");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "CustomNet");

    auto preProc = ov::preprocess::PrePostProcessor(model);
    preProc.input(0).tensor().set_layout("NHWC");
    preProc.input(0).model().set_layout("NCHW");
    preProc.output(0).tensor().set_layout("NHWC");
    preProc.output(0).model().set_layout("NCHW");
    model = preProc.build();

    return model;
}

using InferWithHostCompileMTParams = std::tuple<std::string, ov::AnyMap, std::string>;

class InferWithHostCompileMultithreadTests : public testing::WithParamInterface<InferWithHostCompileMTParams>,
                                             public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferWithHostCompileMTParams> obj) {
        std::string targetDevice;
        ov::AnyMap cfg;
        std::string modelName;
        std::tie(targetDevice, cfg, modelName) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!cfg.empty()) {
            for (auto& configItem : cfg) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
                result << "_";
            }
        }
        result << "model=" << modelName;
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(target_device, configuration, selectedModelName) = this->GetParam();
        if (selectedModelName == "CustomNet") {
            GTEST_SKIP() << "CustomNet is currently skipped for multithread host compile tests";
        }
        configuration[ov::intel_npu::compile_log_level.name()] = ov::log::Level::ERR;

        std::vector<std::string> deviceNames =
            core->get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
        for (const auto& name : deviceNames) {
            if (target_device.find(name) != std::string::npos) {
                isTargetDevice = true;
                break;
            }
        }

        APIBaseTest::SetUp();
    }

    static std::shared_ptr<ov::Model> createModelByName(const std::string& modelName) {
        if (modelName == "CustomNet") {
            return createCustomNetModelMT();
        }
        if (modelName == "MaxPool") {
            return createMaxPoolModelMT();
        }
        if (modelName == "MaxPool_NCHW") {
            return createMaxPoolModelMT(false, false);
        }
        OPENVINO_THROW("Unknown model name for InferWithHostCompileMultithreadTests: ", modelName);
    }

    static void runConcurrently(size_t threadCount,
                                const std::function<void(size_t)>& worker,
                                const std::function<void(const std::shared_future<void>&)>& beforeStart) {
        std::promise<void> startSignal;
        auto start = startSignal.get_future().share();
        std::vector<std::future<void>> futures;
        futures.reserve(threadCount);

        for (size_t i = 0; i < threadCount; ++i) {
            futures.emplace_back(std::async(std::launch::async, [i, &worker, start]() {
                start.wait();
                worker(i);
            }));
        }

        beforeStart(start);
        startSignal.set_value();

        std::vector<std::string> failures;
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                futures[i].get();
            } catch (const std::exception& e) {
                failures.push_back("thread " + std::to_string(i) + ": " + e.what());
            }
        }

        if (!failures.empty()) {
            std::ostringstream os;
            os << "Parallel execution failures (" << failures.size() << "):";
            for (const auto& message : failures) {
                os << "\n" << message;
            }
            FAIL() << os.str();
        }
    }

    static void runConcurrently(size_t threadCount, const std::function<void(size_t)>& worker) {
        runConcurrently(threadCount, worker, [](const std::shared_future<void>&) {});
    }

    static void runConcurrentlyAsync(size_t threadCount,
                                     const std::function<void(size_t)>& prepareInference,
                                     const std::function<void(size_t)>& startInference,
                                     const std::function<void(size_t)>& waitInference) {
        std::mutex mutex;
        std::condition_variable condition;
        size_t readyCount = 0;
        bool start = false;
        bool aborted = false;
        std::vector<std::future<void>> futures;
        futures.reserve(threadCount);

        for (size_t i = 0; i < threadCount; ++i) {
            futures.emplace_back(std::async(std::launch::async,
                                            [i,
                                             &prepareInference,
                                             &startInference,
                                             &waitInference,
                                             &mutex,
                                             &condition,
                                             &readyCount,
                                             &start,
                                             &aborted]() {
                                                try {
                                                    prepareInference(i);
                                                } catch (...) {
                                                    std::lock_guard<std::mutex> lock(mutex);
                                                    aborted = true;
                                                    condition.notify_all();
                                                    throw;
                                                }
                                                {
                                                    std::unique_lock<std::mutex> lock(mutex);
                                                    ++readyCount;
                                                    condition.notify_all();
                                                    condition.wait(lock, [&start, &aborted]() {
                                                        return start || aborted;
                                                    });
                                                    if (aborted) {
                                                        return;
                                                    }
                                                }
                                                startInference(i);
                                                waitInference(i);
                                            }));
        }

        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [&readyCount, threadCount, &aborted]() {
                return readyCount == threadCount || aborted;
            });
            start = true;
        }
        condition.notify_all();

        std::vector<std::string> failures;
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                futures[i].get();
            } catch (const std::exception& e) {
                failures.push_back("thread " + std::to_string(i) + ": " + e.what());
            }
        }

        if (!failures.empty()) {
            std::ostringstream os;
            os << "Parallel async execution failures (" << failures.size() << "):";
            for (const auto& message : failures) {
                os << "\n" << message;
            }
            FAIL() << os.str();
        }
    }

    static void runConcurrentlyAsync(size_t threadCount,
                                     const std::function<void(size_t)>& startInference,
                                     const std::function<void(size_t)>& waitInference) {
        runConcurrentlyAsync(threadCount, [](size_t) {}, startInference, waitInference);
    }

    static ov::Tensor makeInputTensor(const std::shared_ptr<ov::Model>& model,
                                      const ov::Shape& shape,
                                      const int startFrom) {
        return ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, startFrom);
    }

    static void inferAndCompare(ov::InferRequest& reqDynamic, ov::InferRequest& reqReference) {
        reqDynamic.infer();
        reqReference.infer();

        const auto npuOutputTensor = reqDynamic.get_output_tensor(0);
        const auto referenceOutputTensor = reqReference.get_output_tensor(0);
        ov::test::utils::compare(referenceOutputTensor, npuOutputTensor, npuOutputTensor.get_element_type());
    }

    static void waitAndCompare(ov::InferRequest& reqDynamic, ov::InferRequest& reqReference) {
        reqDynamic.wait();
        reqReference.infer();

        const auto npuOutputTensor = reqDynamic.get_output_tensor(0);
        const auto referenceOutputTensor = reqReference.get_output_tensor(0);
        ov::test::utils::compare(referenceOutputTensor, npuOutputTensor, npuOutputTensor.get_element_type());
    }

    static void setInputInferAndCompare(ov::InferRequest& reqDynamic,
                                        ov::InferRequest& reqReference,
                                        const ov::Tensor& inputTensor) {
        reqDynamic.set_input_tensor(0, inputTensor);
        reqReference.set_input_tensor(0, inputTensor);
        inferAndCompare(reqDynamic, reqReference);
    }

    static ov::Tensor makeZeroInputTensor(ov::RemoteContext& zeroContext,
                                          const std::shared_ptr<ov::Model>& model,
                                          const ov::Shape& shape,
                                          const int startFrom) {
        auto zeroInput = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
        auto inputSource = makeInputTensor(model, shape, startFrom);
        if (inputSource.get_byte_size() != zeroInput.get_byte_size()) {
            OPENVINO_THROW("Source and zero input tensors have different byte sizes");
        }
        std::memcpy(zeroInput.data(), inputSource.data(), inputSource.get_byte_size());
        return zeroInput;
    }

protected:
    ov::AnyMap makeConfig(bool sharedQueue) const {
        auto cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;
        return cfg;
    }

    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::string selectedModelName;
    std::string target_device;
    bool isTargetDevice = false;
};

TEST_P(InferWithHostCompileMultithreadTests, MT_PerThreadCompileCreateInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kThreadCount = 4;
    ov::Shape shape;
    if (selectedModelName == "MaxPool_NCHW") {
        shape = {1, 16, 720, 1280};
    } else {
        shape = {1, 720, 1280, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        const auto cfg = makeConfig(sharedQueue);
        std::vector<std::shared_ptr<ov::InferRequest>> requests(kThreadCount);
        std::vector<std::shared_ptr<ov::InferRequest>> referenceRequests(kThreadCount);
        std::vector<ov::Tensor> inputs(kThreadCount);
        runConcurrentlyAsync(
            kThreadCount,
            [this, &cfg, &shape, &referenceCompiledModel, &requests, &referenceRequests, &inputs](size_t threadIdx) {
                auto model = createModelByName(selectedModelName);
                auto compiledModel = core->compile_model(model, target_device, cfg);
                requests[threadIdx] = std::make_shared<ov::InferRequest>(compiledModel.create_infer_request());
                referenceRequests[threadIdx] =
                    std::make_shared<ov::InferRequest>(referenceCompiledModel.create_infer_request());
                inputs[threadIdx] = makeInputTensor(model, shape, static_cast<int>(100 + threadIdx));
                requests[threadIdx]->set_input_tensor(0, inputs[threadIdx]);
                referenceRequests[threadIdx]->set_input_tensor(0, inputs[threadIdx]);
            },
            [&requests](size_t threadIdx) {
                requests[threadIdx]->start_async();
            },
            [&requests, &referenceRequests](size_t threadIdx) {
                waitAndCompare(*requests[threadIdx], *referenceRequests[threadIdx]);
            });
    }
}

TEST_P(InferWithHostCompileMultithreadTests, MT_SingleCompileParallelCreateRequestAndInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kThreadCount = 8;
    ov::Shape shape;
    if (selectedModelName == "MaxPool_NCHW") {
        shape = {1, 16, 720, 1280};
    } else {
        shape = {1, 720, 1280, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        const auto cfg = makeConfig(sharedQueue);
        auto model = createModelByName(selectedModelName);
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, cfg));

        std::vector<std::shared_ptr<ov::InferRequest>> requests(kThreadCount);
        std::vector<std::shared_ptr<ov::InferRequest>> referenceRequests(kThreadCount);
        std::vector<ov::Tensor> inputs(kThreadCount);
        runConcurrentlyAsync(
            kThreadCount,
            [model, &compiledModel, &referenceCompiledModel, &shape, &requests, &referenceRequests, &inputs](
                size_t threadIdx) {
                requests[threadIdx] = std::make_shared<ov::InferRequest>(compiledModel.create_infer_request());
                referenceRequests[threadIdx] =
                    std::make_shared<ov::InferRequest>(referenceCompiledModel.create_infer_request());
                inputs[threadIdx] = makeInputTensor(model, shape, static_cast<int>(100 + threadIdx));
                requests[threadIdx]->set_input_tensor(0, inputs[threadIdx]);
                referenceRequests[threadIdx]->set_input_tensor(0, inputs[threadIdx]);
            },
            [&requests](size_t threadIdx) {
                requests[threadIdx]->start_async();
            },
            [&requests, &referenceRequests](size_t threadIdx) {
                waitAndCompare(*requests[threadIdx], *referenceRequests[threadIdx]);
            });
    }
}

TEST_P(InferWithHostCompileMultithreadTests, MT_ConcurrentInferThenSetPriorityAndInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kThreadCount = 8;
    ov::Shape shape;
    if (selectedModelName == "MaxPool_NCHW") {
        shape = {1, 16, 720, 1280};
    } else {
        shape = {1, 720, 1280, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        auto cfg = makeConfig(sharedQueue);
        cfg[ov::hint::model_priority.name()] = ov::hint::Priority::LOW;

        auto model = createModelByName(selectedModelName);
        ov::CompiledModel compiledModel;
        try {
            compiledModel = core->compile_model(model, target_device, cfg);
        } catch (const ov::Exception& e) {
            GTEST_SKIP() << "model priority is not supported: " << e.what();
        }

        std::vector<ov::InferRequest> requests;
        std::vector<ov::InferRequest> referenceRequests;
        requests.reserve(kThreadCount);
        referenceRequests.reserve(kThreadCount);
        for (size_t i = 0; i < kThreadCount; ++i) {
            requests.emplace_back(compiledModel.create_infer_request());
            referenceRequests.emplace_back(referenceCompiledModel.create_infer_request());
        }

        std::vector<ov::Tensor> firstInputs;
        firstInputs.reserve(kThreadCount);
        for (size_t i = 0; i < kThreadCount; ++i) {
            firstInputs.emplace_back(makeInputTensor(model, shape, static_cast<int>(100 + i)));
            requests[i].set_input_tensor(0, firstInputs.back());
            referenceRequests[i].set_input_tensor(0, firstInputs.back());
        }
        runConcurrentlyAsync(
            kThreadCount,
            [&requests](size_t threadIdx) {
                requests[threadIdx].start_async();
            },
            [&requests, &referenceRequests](size_t threadIdx) {
                waitAndCompare(requests[threadIdx], referenceRequests[threadIdx]);
            });

        OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}}));

        std::vector<ov::Tensor> secondInputs;
        secondInputs.reserve(kThreadCount);
        for (size_t i = 0; i < kThreadCount; ++i) {
            secondInputs.emplace_back(makeInputTensor(model, shape, static_cast<int>(200 + i)));
            requests[i].set_input_tensor(0, secondInputs.back());
            referenceRequests[i].set_input_tensor(0, secondInputs.back());
        }
        runConcurrentlyAsync(
            kThreadCount,
            [&requests](size_t threadIdx) {
                requests[threadIdx].start_async();
            },
            [&requests, &referenceRequests](size_t threadIdx) {
                waitAndCompare(requests[threadIdx], referenceRequests[threadIdx]);
            });

        OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::LOW}}));

        std::vector<ov::Tensor> thirdInputs;
        thirdInputs.reserve(kThreadCount);
        for (size_t i = 0; i < kThreadCount; ++i) {
            thirdInputs.emplace_back(makeInputTensor(model, shape, static_cast<int>(300 + i)));
            requests[i].set_input_tensor(0, thirdInputs.back());
            referenceRequests[i].set_input_tensor(0, thirdInputs.back());
        }
        runConcurrentlyAsync(
            kThreadCount,
            [&requests](size_t threadIdx) {
                requests[threadIdx].start_async();
            },
            [&requests, &referenceRequests](size_t threadIdx) {
                waitAndCompare(requests[threadIdx], referenceRequests[threadIdx]);
            });
    }
}

TEST_P(InferWithHostCompileMultithreadTests, MT_MultiCompiledModelsMultiRequestsMultiInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kThreadCount = 4;
    constexpr size_t kModelCount = 3;
    constexpr size_t kRequestsPerModel = 2;
    constexpr size_t kInferLoops = 3;
    ov::Shape shapeLarge;
    ov::Shape shapeSmall;
    if (selectedModelName == "MaxPool_NCHW") {
        shapeLarge = {1, 16, 720, 1280};
        shapeSmall = {1, 16, 360, 640};
    } else {
        shapeLarge = {1, 720, 1280, 16};
        shapeSmall = {1, 360, 640, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        const auto cfg = makeConfig(sharedQueue);
        auto model = createModelByName(selectedModelName);

        std::vector<ov::CompiledModel> compiledModels;
        compiledModels.reserve(kModelCount);
        auto producer = std::async(std::launch::async, [this, &compiledModels, &model, &cfg, kModelCount]() {
            for (size_t i = 0; i < kModelCount; ++i) {
                compiledModels.emplace_back(core->compile_model(model, target_device, cfg));
            }
        });
        producer.get();

        ASSERT_EQ(compiledModels.size(), kModelCount);

        std::atomic<size_t> successInferCount{0};
        std::vector<std::vector<ov::InferRequest>> requests(kThreadCount);
        std::vector<std::vector<ov::InferRequest>> referenceRequests(kThreadCount);
        std::vector<std::vector<ov::Tensor>> inputs(kThreadCount);
        for (size_t inferIdx = 0; inferIdx < kInferLoops; ++inferIdx) {
            const bool useAltShape = (inferIdx % 2U) == 1U;
            const auto& shape = useAltShape ? shapeSmall : shapeLarge;
            runConcurrentlyAsync(
                kThreadCount,
                [model,
                 &compiledModels,
                 &referenceCompiledModel,
                 &requests,
                 &referenceRequests,
                 &inputs,
                 &shape,
                 kRequestsPerModel,
                 inferIdx](size_t threadIdx) {
                    if (requests[threadIdx].empty()) {
                        requests[threadIdx].reserve(compiledModels.size() * kRequestsPerModel);
                        referenceRequests[threadIdx].reserve(compiledModels.size() * kRequestsPerModel);
                    }
                    inputs[threadIdx].clear();
                    inputs[threadIdx].reserve(compiledModels.size() * kRequestsPerModel);
                    for (size_t modelIdx = 0; modelIdx < compiledModels.size(); ++modelIdx) {
                        for (size_t reqIdx = 0; reqIdx < kRequestsPerModel; ++reqIdx) {
                            if (inferIdx == 0) {
                                requests[threadIdx].emplace_back(compiledModels[modelIdx].create_infer_request());
                                referenceRequests[threadIdx].emplace_back(
                                    referenceCompiledModel.create_infer_request());
                            }
                            const int startFrom =
                                static_cast<int>(100 + threadIdx * 17 + modelIdx * 7 + reqIdx * 3 + inferIdx);
                            inputs[threadIdx].emplace_back(makeInputTensor(model, shape, startFrom));
                            const size_t requestIdx = modelIdx * kRequestsPerModel + reqIdx;
                            requests[threadIdx][requestIdx].set_input_tensor(0, inputs[threadIdx].back());
                            referenceRequests[threadIdx][requestIdx].set_input_tensor(0, inputs[threadIdx].back());
                        }
                    }
                },
                [&requests](size_t threadIdx) {
                    for (auto& request : requests[threadIdx]) {
                        request.start_async();
                    }
                },
                [&requests, &referenceRequests, &successInferCount](size_t threadIdx) {
                    for (size_t requestIdx = 0; requestIdx < requests[threadIdx].size(); ++requestIdx) {
                        waitAndCompare(requests[threadIdx][requestIdx], referenceRequests[threadIdx][requestIdx]);
                        successInferCount.fetch_add(1, std::memory_order_relaxed);
                    }
                });
        }

        const size_t expectedInferCount = kThreadCount * kModelCount * kRequestsPerModel * kInferLoops;
        ASSERT_EQ(successInferCount.load(std::memory_order_relaxed), expectedInferCount)
            << "Unexpected total successful infer count";
    }
}

TEST_P(InferWithHostCompileMultithreadTests, MT_SingleCompileParallelZeroInputOutputTensorInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kThreadCount = 4;
    constexpr size_t kInferLoops = 3;
    ov::Shape shapeLarge;
    ov::Shape shapeSmall;
    if (selectedModelName == "MaxPool_NCHW") {
        shapeLarge = {1, 16, 720, 1280};
        shapeSmall = {1, 16, 360, 640};
    } else {
        shapeLarge = {1, 720, 1280, 16};
        shapeSmall = {1, 360, 640, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        const auto cfg = makeConfig(sharedQueue);
        auto model = createModelByName(selectedModelName);
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, cfg));

        std::atomic<size_t> successInferCount{0};
        std::vector<std::shared_ptr<ov::InferRequest>> requests(kThreadCount);
        std::vector<std::shared_ptr<ov::InferRequest>> referenceRequests(kThreadCount);
        std::vector<ov::Tensor> inputs(kThreadCount);
        std::vector<ov::Tensor> outputs(kThreadCount);
        for (size_t inferIdx = 0; inferIdx < kInferLoops; ++inferIdx) {
            const bool useAltShape = (inferIdx % 2U) == 1U;
            const auto& shape = useAltShape ? shapeSmall : shapeLarge;
            runConcurrentlyAsync(
                kThreadCount,
                [this,
                 model,
                 &compiledModel,
                 &referenceCompiledModel,
                 &requests,
                 &referenceRequests,
                 &inputs,
                 &outputs,
                 &shape,
                 inferIdx](size_t threadIdx) {
                    if (!requests[threadIdx]) {
                        requests[threadIdx] = std::make_shared<ov::InferRequest>(compiledModel.create_infer_request());
                        referenceRequests[threadIdx] =
                            std::make_shared<ov::InferRequest>(referenceCompiledModel.create_infer_request());
                    }
                    auto zeroContext = core->get_default_context(target_device);
                    const int startFrom = static_cast<int>(100 + threadIdx * 11 + inferIdx);
                    inputs[threadIdx] = makeZeroInputTensor(zeroContext, model, shape, startFrom);
                    outputs[threadIdx] = zeroContext.create_host_tensor(model->output().get_element_type(), shape);
                    requests[threadIdx]->set_tensor(model->input(), inputs[threadIdx]);
                    requests[threadIdx]->set_tensor(model->output(), outputs[threadIdx]);
                    referenceRequests[threadIdx]->set_input_tensor(0, inputs[threadIdx]);
                },
                [&requests](size_t threadIdx) {
                    requests[threadIdx]->start_async();
                },
                [&requests, &referenceRequests, &successInferCount](size_t threadIdx) {
                    waitAndCompare(*requests[threadIdx], *referenceRequests[threadIdx]);
                    successInferCount.fetch_add(1, std::memory_order_relaxed);
                });
        }

        ASSERT_EQ(successInferCount.load(std::memory_order_relaxed), kThreadCount * kInferLoops)
            << "Unexpected successful zero tensor infer count";
    }
}

TEST_P(InferWithHostCompileMultithreadTests, MT_PerThreadCompileZeroInputOutputTensorInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kThreadCount = 4;
    constexpr size_t kInferLoops = 3;
    ov::Shape shapeLarge;
    ov::Shape shapeSmall;
    if (selectedModelName == "MaxPool_NCHW") {
        shapeLarge = {1, 16, 720, 1280};
        shapeSmall = {1, 16, 360, 640};
    } else {
        shapeLarge = {1, 720, 1280, 16};
        shapeSmall = {1, 360, 640, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        const auto cfg = makeConfig(sharedQueue);
        std::atomic<size_t> successInferCount{0};
        std::vector<std::shared_ptr<ov::Model>> models(kThreadCount);
        std::vector<std::shared_ptr<ov::CompiledModel>> compiledModels(kThreadCount);
        std::vector<std::shared_ptr<ov::InferRequest>> requests(kThreadCount);
        std::vector<std::shared_ptr<ov::InferRequest>> referenceRequests(kThreadCount);
        std::vector<ov::Tensor> inputs(kThreadCount);
        std::vector<ov::Tensor> outputs(kThreadCount);
        for (size_t inferIdx = 0; inferIdx < kInferLoops; ++inferIdx) {
            const bool useAltShape = (inferIdx % 2U) == 1U;
            const auto& shape = useAltShape ? shapeSmall : shapeLarge;
            runConcurrentlyAsync(
                kThreadCount,
                [this,
                 &cfg,
                 &referenceCompiledModel,
                 &models,
                 &compiledModels,
                 &requests,
                 &referenceRequests,
                 &inputs,
                 &outputs,
                 &shape,
                 inferIdx](size_t threadIdx) {
                    if (!models[threadIdx]) {
                        models[threadIdx] = createModelByName(selectedModelName);
                        compiledModels[threadIdx] = std::make_shared<ov::CompiledModel>(
                            core->compile_model(models[threadIdx], target_device, cfg));
                        requests[threadIdx] =
                            std::make_shared<ov::InferRequest>(compiledModels[threadIdx]->create_infer_request());
                        referenceRequests[threadIdx] =
                            std::make_shared<ov::InferRequest>(referenceCompiledModel.create_infer_request());
                    }
                    auto zeroContext = core->get_default_context(target_device);
                    const int startFrom = static_cast<int>(100 + threadIdx * 11 + inferIdx);
                    inputs[threadIdx] = makeZeroInputTensor(zeroContext, models[threadIdx], shape, startFrom);
                    outputs[threadIdx] =
                        zeroContext.create_host_tensor(models[threadIdx]->output().get_element_type(), shape);
                    requests[threadIdx]->set_tensor(models[threadIdx]->input(), inputs[threadIdx]);
                    requests[threadIdx]->set_tensor(models[threadIdx]->output(), outputs[threadIdx]);
                    referenceRequests[threadIdx]->set_input_tensor(0, inputs[threadIdx]);
                },
                [&requests](size_t threadIdx) {
                    requests[threadIdx]->start_async();
                },
                [&requests, &referenceRequests, &successInferCount](size_t threadIdx) {
                    waitAndCompare(*requests[threadIdx], *referenceRequests[threadIdx]);
                    successInferCount.fetch_add(1, std::memory_order_relaxed);
                });
        }

        ASSERT_EQ(successInferCount.load(std::memory_order_relaxed), kThreadCount * kInferLoops)
            << "Unexpected successful per-thread zero tensor infer count";
    }
}

TEST_P(InferWithHostCompileMultithreadTests, MT_CompileAndInferOverlap) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    constexpr size_t kModelCount = 4;
    constexpr size_t kThreadCount = 4;
    ov::Shape shape;
    if (selectedModelName == "MaxPool_NCHW") {
        shape = {1, 16, 720, 1280};
    } else {
        shape = {1, 720, 1280, 16};
    }

    auto referenceModel = createModelByName(selectedModelName);
    ov::CompiledModel referenceCompiledModel;
    try {
        referenceCompiledModel = core->compile_model(referenceModel, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    for (bool sharedQueue : {true, false}) {
        const auto cfg = makeConfig(sharedQueue);
        auto model = createModelByName(selectedModelName);

        std::vector<ov::CompiledModel> compiledModels;
        compiledModels.reserve(kModelCount);
        std::mutex mutex;
        std::condition_variable cv;
        bool producerDone = false;
        std::exception_ptr producerException;

        std::atomic<size_t> nextModelIdx{0};
        std::atomic<size_t> successInferCount{0};
        std::future<void> producer;
        runConcurrently(
            kThreadCount,
            [model,
             &referenceCompiledModel,
             &shape,
             &compiledModels,
             &mutex,
             &cv,
             &producerDone,
             &producerException,
             &nextModelIdx,
             &successInferCount](size_t threadIdx) {
                while (true) {
                    ov::CompiledModel compiledModel;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [&compiledModels, &producerDone, &nextModelIdx]() {
                            return nextModelIdx.load(std::memory_order_relaxed) < compiledModels.size() || producerDone;
                        });
                        if (producerException) {
                            std::rethrow_exception(producerException);
                        }
                        const size_t idx = nextModelIdx.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= compiledModels.size()) {
                            break;
                        }
                        compiledModel = compiledModels[idx];
                    }

                    auto reqDynamic = compiledModel.create_infer_request();
                    auto reqReference = referenceCompiledModel.create_infer_request();
                    auto input = makeInputTensor(model, shape, static_cast<int>(100 + threadIdx));
                    reqDynamic.set_input_tensor(0, input);
                    reqReference.set_input_tensor(0, input);
                    reqDynamic.start_async();
                    waitAndCompare(reqDynamic, reqReference);
                    successInferCount.fetch_add(1, std::memory_order_relaxed);
                }
            },
            [this,
             &cfg,
             &model,
             &compiledModels,
             &mutex,
             &cv,
             &producerDone,
             &producerException,
             kModelCount,
             &producer](const std::shared_future<void>& start) {
                producer = std::async(std::launch::async,
                                      [this,
                                       &cfg,
                                       &model,
                                       &compiledModels,
                                       &mutex,
                                       &cv,
                                       &producerDone,
                                       &producerException,
                                       start,
                                       kModelCount]() {
                                          start.wait();
                                          try {
                                              for (size_t i = 0; i < kModelCount; ++i) {
                                                  auto compiledModel = core->compile_model(model, target_device, cfg);
                                                  {
                                                      std::lock_guard<std::mutex> lock(mutex);
                                                      compiledModels.push_back(compiledModel);
                                                  }
                                                  cv.notify_all();
                                              }
                                          } catch (...) {
                                              std::lock_guard<std::mutex> lock(mutex);
                                              producerException = std::current_exception();
                                              producerDone = true;
                                              cv.notify_all();
                                              return;
                                          }
                                          {
                                              std::lock_guard<std::mutex> lock(mutex);
                                              producerDone = true;
                                          }
                                          cv.notify_all();
                                      });
            });
        producer.get();

        ASSERT_EQ(successInferCount.load(std::memory_order_relaxed), kModelCount)
            << "Each compiled model should be consumed by one overlapping infer worker";
    }
}

const std::vector<std::string> mtDevices = {"NPU.4000", "NPU.5010"};

const std::vector<ov::AnyMap> mtConfigs = {
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"},
        {"NPU_CREATE_EXECUTOR", "0"},
    },
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"},
    },
};

const std::vector<std::string> mtModelNames = {"MaxPool_NCHW" /*, "CustomNet"*/};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferWithHostCompileMultithreadTests,
                         ::testing::Combine(::testing::ValuesIn(mtDevices),
                                            ::testing::ValuesIn(mtConfigs),
                                            ::testing::ValuesIn(mtModelNames)),
                         InferWithHostCompileMultithreadTests::getTestCaseName);

}  // namespace behavior
}  // namespace test
}  // namespace ov