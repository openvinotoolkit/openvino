// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>

#include <gtest/gtest.h>
// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "gna_groups.hpp"

namespace {
std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> input_shapes_2d {
    {{1, 128}, {8, 16}},
    {{1, 64}, {8, 8}},
    {{1, 56}, {7, 8}},
    {{1, 48}, {6, 8}},
    {{1, 40}, {5, 8}},
    {{1, 32}, {4, 8}},
    {{1, 24}, {3, 8}},
    {{1, 16}, {2, 8}},
    {{1, 8}, {1, 8}},
    {{1, 19}, {1, 19}},
    {{128, 1}, {8, 16}},
    {{64, 1}, {8, 8}},
    {{56, 1}, {7, 8}},
    {{48, 1}, {6, 8}},
    {{40, 1}, {5, 8}},
    {{32, 1}, {4, 8}},
    {{24, 1}, {3, 8}},
    {{16, 1}, {2, 8}},
    {{8, 1}, {1, 8}},
    {{19, 1}, {1, 19}}
};

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> input_shapes_4d {
    {{1, 2, 2, 32}, {8, 16, 1, 1}},
    {{1, 2, 4, 8}, {8, 8, 1, 1}},
    {{1, 2, 2, 14}, {7, 8, 1, 1}},
    {{1, 2, 4, 6}, {6, 8, 1, 1}},
    {{1, 2, 2, 10}, {5, 8, 1, 1}},
    {{1, 2, 2, 8}, {4, 8, 1, 1}},
    {{1, 2, 2, 6}, {3, 8, 1, 1}},
    {{1, 2, 2, 4}, {2, 8, 1, 1}},
    {{1, 2, 2, 2}, {1, 8, 1, 1}},
    {{1, 1, 1, 19}, {1, 19, 1, 1}},
    {{32, 2, 2, 1}, {8, 16, 1, 1}},
    {{8, 4, 2, 1}, {8, 8, 1, 1}},
    {{14, 2, 2, 1}, {7, 8, 1, 1}},
    {{6, 4, 2, 1}, {6, 8, 1, 1}},
    {{10, 2, 2, 1}, {5, 8, 1, 1}},
    {{8, 2, 2, 1}, {4, 8, 1, 1}},
    {{6, 2, 2, 1}, {3, 8, 1, 1}},
    {{4, 2, 2, 1}, {2, 8, 1, 1}},
    {{2, 2, 2, 1}, {1, 8, 1, 1}},
    {{19, 1, 1, 1}, {1, 19, 1, 1}}
};

class Get2DReshapedDataTest : public ::testing::Test {
 protected:
    const char* input_name = "input";
    const InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    const size_t max_batch_size = 8;
    void Reshape2dAndCheck(const std::pair<std::vector<size_t>, std::vector<size_t>>& input_shape,
                           InferenceEngine::Layout layout) const {
        auto data = std::make_shared<InferenceEngine::Data>(input_name,
            InferenceEngine::TensorDesc(precision, input_shape.first, layout));
        auto new_data = GNAPluginNS::Get2DReshapedData(data, max_batch_size);
        ASSERT_EQ(new_data->getDims(), input_shape.second);
        ASSERT_EQ(new_data->getPrecision(), precision);
        ASSERT_EQ(new_data->getLayout(), layout);
    }
};

TEST_F(Get2DReshapedDataTest, testReshape2D) {
    auto layout = InferenceEngine::NC;
    for (const auto &input_shape : input_shapes_2d) {
        Reshape2dAndCheck(input_shape, layout);
    }
}

TEST_F(Get2DReshapedDataTest, testReshape4D) {
    auto layout = InferenceEngine::NCHW;
    for (const auto &input_shape : input_shapes_4d) {
        Reshape2dAndCheck(input_shape, layout);
    }
}
} // namespace