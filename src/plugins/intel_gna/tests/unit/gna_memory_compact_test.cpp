// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <legacy/details/ie_cnn_network_tools.h>
#include <legacy/ie_layers.h>

#include <legacy/graph_tools.hpp>
#include <memory>
#include <vector>

#include "gna_data_types.hpp"
#include "gna_fused_iterator.hpp"
#include "gna_plugin.hpp"
#include "memory/gna_memory.hpp"
#include "ov_models/builders.hpp"

using namespace InferenceEngine;
using namespace memory;

class GNAMemoryCompactTest : public ::testing::Test {
protected:
    GNAMemory<memory::GNAFloatAllocator> mem;
    bool isCompact = true;

    void SetUp() override {}
};

TEST_F(GNAMemoryCompactTest, canOptimizeReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    auto scratch = mem.getQueue(rRegion::REGION_SCRATCH);
    scratch->reserve_ptr(layer1, pFuture1, 3 * sizeof(float));
    scratch->reserve_ptr(layer2, pFuture2, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(scratch->getSize(), 3 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizePushValue) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    auto scratch = mem.getQueue(rRegion::REGION_SCRATCH);
    scratch->push_value(layer1, pFuture1, 1.f, 2);
    scratch->push_value(layer2, pFuture2, 2.f, 3);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_SCRATCH), 5 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizePushValueAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    scratchQueue->push_value(layer1, pFuture1, 3.f, 2);
    scratchQueue->bind_ptr(layer2, pFuture2, pFuture1, 0, 2);
    scratchQueue->reserve_ptr(layer3, pFuture3, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(scratchQueue->getSize(), 2 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizeTwoPushValueAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    CNNLayerPtr layer4 = std::make_shared<CNNLayer>(LayerParams("layer4", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    layer4->userValue.v_int = 4;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    scratchQueue->push_value(layer1, pFuture1, 1.f, 2);
    scratchQueue->push_value(layer2, pFuture2, 2.f, 3);
    scratchQueue->reserve_ptr(layer3, pFuture3, 5 * sizeof(float));
    scratchQueue->bind_ptr(layer2, pFuture2, pFuture1, 0, 2);

    mem.commit(isCompact);
    ASSERT_EQ(scratchQueue->getSize(), 5 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizePushPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float input[] = {1, 2, 3};
    size_t input_size = sizeof(input);

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    scratchQueue->push_ptr(layer1, pFuture1, input, input_size);
    scratchQueue->reserve_ptr(layer2, pFuture2, input_size);
    scratchQueue->bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(scratchQueue->getSize(), input_size);
}

TEST_F(GNAMemoryCompactTest, canOptimizePushLocalPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    size_t input_size;
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        input_size = input.size() * sizeof(float);
        scratchQueue->push_local_ptr(layer1, pFuture1, &*input.begin(), input_size);
    }

    scratchQueue->reserve_ptr(layer2, pFuture2, input_size);
    scratchQueue->bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(scratchQueue->getSize(), input_size);
}

TEST_F(GNAMemoryCompactTest, canOptimizePushInitilizerPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    size_t input_size;
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        input_size = input.size() * sizeof(float);
        scratchQueue->push_initializer(layer1, pFuture1, input_size, [=](void* data, size_t size) {
            ie_memcpy(data, size, &input[0], input.size());
        });
    }

    scratchQueue->reserve_ptr(layer2, pFuture2, 2 * input_size);
    scratchQueue->bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(scratchQueue->getSize(), 2 * input_size);
}

TEST_F(GNAMemoryCompactTest, canOptimizeBindInitilizerPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    CNNLayerPtr layer4 = std::make_shared<CNNLayer>(LayerParams("layer4", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    layer4->userValue.v_int = 4;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);
    float* pFuture4 = reinterpret_cast<float*>(&pFuture4);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        scratchQueue->bind_initializer(layer2, pFuture1, [=](void* data, size_t size) {
            ie_memcpy(data, size, &input[0], input.size());
        });
    }

    scratchQueue->reserve_ptr(layer1, pFuture1, 4 * sizeof(float));
    scratchQueue->reserve_ptr(layer3, pFuture3, 2 * sizeof(float));
    scratchQueue->bind_ptr(layer4, pFuture4, pFuture3, 0, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(scratchQueue->getSize(), 4 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizeReservePtrWithOffset) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    auto scratchQueue = mem.getQueue(rRegion::REGION_SCRATCH);
    scratchQueue->reserve_ptr(layer1, pFuture1, 2 * sizeof(float));
    scratchQueue->reserve_ptr(layer2, pFuture2, 2 * sizeof(float));
    scratchQueue->bind_ptr(layer3, pFuture3, pFuture2, 2 * sizeof(float), 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_SCRATCH), 4 * sizeof(float));
}

class GNAMemoryTested : public memory::GNAMemory<memory::GNAFloatAllocator> {
    using GNAMemory::GNAMemory;

public:
    void Test() {
        // filtering RW allocation requests only
        auto filter_req = [](const MemRequest& re) {
            return re._region == REGION_SCRATCH && re._type != REQUEST_BIND;
        };
        std::vector<MemRequest> test_reqs;
        const auto& requests = getQueue(REGION_SCRATCH)->_mem_requests;

        auto it = std::copy_if(requests.begin(), requests.end(), std::back_inserter(test_reqs), filter_req);

        // intercrossing condition
        auto is_crossed = [](const MemRequest& re1, const MemRequest& re2) {
            const std::pair<uint16_t, uint16_t> limits_default{0, UINT16_MAX};
            if (re1._life_limits == limits_default || re2._life_limits == limits_default) {
                return true;
            }
            return (re1._life_limits.first > re2._life_limits.first &&
                    re1._life_limits.first < re2._life_limits.second) ||
                   (re1._life_limits.second > re2._life_limits.first &&
                    re1._life_limits.second < re2._life_limits.second);
        };

        // verify that requests are intercrossed
        for (auto re_it_1 = test_reqs.begin(); re_it_1 != test_reqs.end(); ++re_it_1) {
            for (auto re_it_2 = re_it_1 + 1; re_it_2 != test_reqs.end(); ++re_it_2) {
                ASSERT_TRUE(is_crossed(*re_it_1, *re_it_2));
            }
        }
    }
};

class GNAPluginTested : public GNAPlugin {
public:
    std::shared_ptr<GNAMemoryTested> gnamem_t;
    GNAPluginTested() : GNAPlugin() {
        gnamem_t = std::make_shared<GNAMemoryTested>();
        gnamem = gnamem_t;
        m_graph_compiler->setGNAMemoryPtr(gnamem);
        gnadevice.reset();
    }
    void Test() {
        gnamem_t->Test();
    }
};

class GNAMemoryOrderTest : public ::testing::Test {};

TEST_F(GNAMemoryOrderTest, orderingFusedLayersActivation) {
    auto plugin = GNAPluginTested();

    ov::Shape input_shape = {1, 16, 20, 16};
    ov::Strides strides = {1, 1};
    ov::Strides dilations = {1, 1};
    ov::CoordinateDiff pad_begin(0, 0), pad_end(0, 0);
    auto weights = ngraph::builder::makeConstant<float>(ov::element::f32, {8, 16, 1, 1}, {1.f});

    auto input = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, input_shape);
    auto conv = std::make_shared<ngraph::opset8::Convolution>(input, weights, strides, pad_begin, pad_end, dilations);
    auto activation =
        ngraph::builder::makeActivation(conv, ov::element::f32, ngraph::helpers::ActivationTypes::Sigmoid);
    auto result = std::make_shared<ngraph::opset8::Result>(activation);
    auto function =
        std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "convolution");

    InferenceEngine::CNNNetwork cnn_network(function);
    plugin.LoadNetwork(cnn_network);
    plugin.Test();
}

TEST_F(GNAMemoryOrderTest, orderingFusedLayersMaxPool) {
    auto plugin = GNAPluginTested();

    ov::Shape input_shape = {1, 16, 20, 16};
    ov::Strides strides = {1, 1};
    ov::Strides dilations = {1, 1};
    ov::CoordinateDiff pad_begin(0, 0), pad_end(0, 0);
    auto weights = ngraph::builder::makeConstant<float>(ov::element::f32, {8, 16, 1, 1}, {1.f});

    auto input = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, input_shape);
    auto conv = std::make_shared<ngraph::opset8::Convolution>(input, weights, strides, pad_begin, pad_end, dilations);
    auto maxpool = ngraph::builder::makePooling(conv,
                                                {1, 1},
                                                {0, 0},
                                                {0, 0},
                                                {1, 1},
                                                ngraph::op::RoundingType::FLOOR,
                                                ngraph::op::PadType::VALID,
                                                false,
                                                ngraph::helpers::PoolingTypes::MAX);
    auto result = std::make_shared<ngraph::opset8::Result>(maxpool);
    auto function =
        std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "convolution");

    InferenceEngine::CNNNetwork cnn_network(function);
    plugin.LoadNetwork(cnn_network);
    plugin.Test();
}

TEST_F(GNAMemoryOrderTest, orderingFusedLayersActivationMaxPool) {
    auto plugin = GNAPluginTested();

    ov::Shape input_shape = {1, 16, 20, 16};
    ov::Strides strides = {1, 1};
    ov::Strides dilations = {1, 1};
    ov::CoordinateDiff pad_begin(0, 0), pad_end(0, 0);
    auto weights = ngraph::builder::makeConstant<float>(ov::element::f32, {8, 16, 1, 1}, {1.f});

    auto input = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, input_shape);
    auto conv = std::make_shared<ngraph::opset8::Convolution>(input, weights, strides, pad_begin, pad_end, dilations);
    auto activation =
        ngraph::builder::makeActivation(conv, ov::element::f32, ngraph::helpers::ActivationTypes::Sigmoid);
    auto maxpool = ngraph::builder::makePooling(activation,
                                                {1, 1},
                                                {0, 0},
                                                {0, 0},
                                                {1, 1},
                                                ngraph::op::RoundingType::FLOOR,
                                                ngraph::op::PadType::VALID,
                                                false,
                                                ngraph::helpers::PoolingTypes::MAX);
    auto result = std::make_shared<ngraph::opset8::Result>(maxpool);
    auto function =
        std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "convolution");

    InferenceEngine::CNNNetwork cnn_network(function);
    plugin.LoadNetwork(cnn_network);
    plugin.Test();
}
