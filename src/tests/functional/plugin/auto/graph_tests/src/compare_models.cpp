// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "graph_tests/include/infer_consistency_test.hpp"

using namespace SubgraphTestsDefinitions;

static std::vector<std::string> modelPaths = {
        std::string("alexnet/FP32/alexnet.xml"),
        std::string("alexnet/FP16/alexnet.xml"),
        std::string("faster_rcnn_inception_resnet_v2_atrous_coco/FP32/faster_rcnn_inception_resnet_v2_atrous_coco.xml"),
        std::string("faster_rcnn_inception_resnet_v2_atrous_coco/FP16/faster_rcnn_inception_resnet_v2_atrous_coco.xml"),
        std::string("faster_rcnn_resnet50_coco/FP32/faster_rcnn_resnet50_coco.xml"),
        std::string("faster_rcnn_resnet50_coco/FP16/faster_rcnn_resnet50_coco.xml"),
        std::string("mobilenet-v1-1.0-224-tf/FP32/mobilenet-v1-1.0-224-tf.xml"),
        std::string("mobilenet-v1-1.0-224-tf/FP16/mobilenet-v1-1.0-224-tf.xml"),
        std::string("resnet-50-tf/FP32/resnet-50-tf.xml"),
        std::string("resnet-50-tf/FP16/resnet-50-tf.xml"),
        std::string("ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml"),
        std::string("ssd_mobilenet_v1_coco/FP16/ssd_mobilenet_v1_coco.xml"),
        std::string("ssd-resnet34-1200-onnx/FP32/ssd-resnet34-1200-onnx.xml"),
        std::string("ssd-resnet34-1200-onnx/FP16/ssd-resnet34-1200-onnx.xml"),
        std::string("vgg16/FP32/vgg16.xml"),
        std::string("vgg16/FP16/vgg16.xml"),
        std::string("vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.xml"),
        std::string("vehicle-detection-adas-0002/FP16/vehicle-detection-adas-0002.xml"),
};


namespace AutoInferThroughput {
    std::map<std::string, std::string>  baseDeviceConfig = {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(THROUGHPUT)}};
    std::map<std::string, std::string>  autoDeviceConfig = {baseDeviceConfig.begin(), baseDeviceConfig.end()};

    const auto param50InferThroughputCPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferThroughputGPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferThroughputCPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("AUTO:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferThroughputGPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("AUTO:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));

    INSTANTIATE_TEST_SUITE_P(AutoCPUTestNightly, AutoInferConsistency, param50InferThroughputCPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoGPUTestNightly, AutoInferConsistency, param50InferThroughputGPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoCPUTest, AutoInferConsistency, param50InferThroughputCPU, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoGPUTest, AutoInferConsistency, param50InferThroughputGPU, AutoInferConsistency::getTestCaseName);
} // namespace AutoInferThroughput

namespace AutoInferLatency {
    std::map<std::string, std::string>  baseDeviceConfig = {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}};
    std::map<std::string, std::string>  autoDeviceConfig = {baseDeviceConfig.begin(), baseDeviceConfig.end()};
    const auto param50InferLatencyCPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferLatencyGPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferLatencyCPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("AUTO:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferLatencyGPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("AUTO:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    INSTANTIATE_TEST_SUITE_P(AutoCPUTestNightly, AutoInferConsistency, param50InferLatencyCPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoGPUTestNightly, AutoInferConsistency, param50InferLatencyGPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoCPUTest, AutoInferConsistency, param50InferLatencyCPU, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoGPUTest, AutoInferConsistency, param50InferLatencyGPU, AutoInferConsistency::getTestCaseName);
} // namespace AutoInferLatency

namespace MultiInferThroughput {
    std::map<std::string, std::string>  baseDeviceConfig = {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(THROUGHPUT)}};
    std::map<std::string, std::string>  autoDeviceConfig = {baseDeviceConfig.begin(), baseDeviceConfig.end()};

    const auto param50InferThroughputCPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("MULTI:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferThroughputGPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("MULTI:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferThroughputCPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("MULTI:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferThroughputGPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("MULTI:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));

    INSTANTIATE_TEST_SUITE_P(MultiCPUTestNightly, AutoInferConsistency, param50InferThroughputCPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(MultiGPUTestNightly, AutoInferConsistency, param50InferThroughputGPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(MultiCPUTest, AutoInferConsistency, param50InferThroughputCPU, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(MultiGPUTest, AutoInferConsistency, param50InferThroughputGPU, AutoInferConsistency::getTestCaseName);
} // namespace MultiInferThroughput

namespace MultiInferLatency {
    std::map<std::string, std::string>  baseDeviceConfig = {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}};
    std::map<std::string, std::string>  autoDeviceConfig = {baseDeviceConfig.begin(), baseDeviceConfig.end()};
    const auto param50InferLatencyCPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("MULTI:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferLatencyGPUNightly = ::testing::Combine(
            ::testing::Values(50),
            ::testing::Values(std::string()),
            ::testing::Values("MULTI:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferLatencyCPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("MULTI:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param50InferLatencyGPU = ::testing::Combine(
            ::testing::Values(50),
            ::testing::ValuesIn(modelPaths),
            ::testing::Values("MULTI:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    INSTANTIATE_TEST_SUITE_P(MultiCPUTestNightly, AutoInferConsistency, param50InferLatencyCPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(MultiGPUTestNightly, AutoInferConsistency, param50InferLatencyGPUNightly, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(MultiCPUTest, AutoInferConsistency, param50InferLatencyCPU, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(MultiGPUTest, AutoInferConsistency, param50InferLatencyGPU, AutoInferConsistency::getTestCaseName);
} // namespace MultiInferLatency
