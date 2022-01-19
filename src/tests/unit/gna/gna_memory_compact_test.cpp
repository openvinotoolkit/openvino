// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <legacy/ie_layers.h>
#include "memory/gna_memory.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS::memory;

class GNAMemoryCompactTest : public ::testing::Test {
 protected:
    GNAMemory<std::allocator<uint8_t>> mem;
    bool isCompact = true;

    void SetUp() override  {
    }
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

    mem.reserve_ptr(layer1, pFuture1, 3 * sizeof(float));
    mem.reserve_ptr(layer2, pFuture2, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 3 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 3 * sizeof(float));
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

    mem.push_value(layer1, pFuture1, 1.f, 2);
    mem.push_value(layer2, pFuture2, 2.f, 3);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 5 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 5 * sizeof(float));
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

    mem.push_value(layer1, pFuture1, 3.f, 2);
    mem.bind_ptr(layer2, pFuture2, pFuture1, 0, 2);
    mem.reserve_ptr(layer3, pFuture3, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 2 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 2 * sizeof(float));
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

    mem.push_value(layer1, pFuture1, 1.f, 2);
    mem.push_value(layer2, pFuture2, 2.f, 3);
    mem.reserve_ptr(layer3, pFuture3, 5 * sizeof(float));
    mem.bind_ptr(layer2, pFuture2, pFuture1, 0, 2);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 5 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 5 * sizeof(float));
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

    float input[]  = {1, 2, 3};
    size_t input_size = sizeof(input);

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    mem.push_ptr(layer1, pFuture1, input, input_size);
    mem.reserve_ptr(layer2, pFuture2, input_size);
    mem.bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), input_size);
    ASSERT_EQ(mem.getTotalBytes(), input_size);
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

    size_t input_size;
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        input_size = input.size() * sizeof(float);
        mem.push_local_ptr(layer1, pFuture1, &*input.begin(), input_size);
    }

    mem.reserve_ptr(layer2, pFuture2, input_size);
    mem.bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), input_size);
    ASSERT_EQ(mem.getTotalBytes(), input_size);
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

    size_t input_size;
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        input_size = input.size() * sizeof(float);
        mem.push_initializer(layer1, pFuture1, input_size, [=](void* data, size_t size){
            ie_memcpy(data, size, &input[0], input.size());
        });
    }

    mem.reserve_ptr(layer2, pFuture2, 2 * input_size);
    mem.bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 2 * input_size);
    ASSERT_EQ(mem.getTotalBytes(), 2 * input_size);
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

    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        mem.bind_initializer(layer2, pFuture1, [=](void* data, size_t size){
            ie_memcpy(data, size, &input[0], input.size());
        });
    }

    mem.reserve_ptr(layer1, pFuture1, 4 * sizeof(float));
    mem.reserve_ptr(layer3, pFuture3, 2 * sizeof(float));
    mem.bind_ptr(layer4, pFuture4, pFuture3, 0, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 4 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 4 * sizeof(float));
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

    mem.reserve_ptr(layer1, pFuture1, 2 * sizeof(float));
    mem.reserve_ptr(layer2, pFuture2, 2 * sizeof(float));
    mem.bind_ptr(layer3, pFuture3, pFuture2, 2 * sizeof(float), 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 4 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 4 * sizeof(float));
}