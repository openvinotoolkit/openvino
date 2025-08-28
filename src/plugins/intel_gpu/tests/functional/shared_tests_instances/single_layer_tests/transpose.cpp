// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/transpose.hpp"

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/single_op/transpose.hpp"

namespace {
using ov::test::TransposeLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
};

/**
 * 4D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{1, 3, 100, 100}},
        // use permute_8x8_4x4 kernel
        {{2, 8, 64, 64}},
        {{2, 5, 64, 64}},
        {{2, 8, 64, 5}},
        {{2, 5, 64, 5}},
};

const std::vector<std::vector<size_t>> inputOrder = {
        // use permute_ref
        std::vector<size_t>{0, 3, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

/**
 * 5D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes5D = {
        {{2, 3, 4, 12, 64}},
        {{2, 5, 11, 32, 32}},
        {{2, 8, 64, 32, 5}},
        {{2, 5, 64, 32, 5}},
};

const std::vector<std::vector<size_t>> inputOrder5D = {
        // use permute_ref
        std::vector<size_t>{0, 3, 4, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 4, 1},
        // use permute_kernel_bfzyx_bfyxz
        std::vector<size_t>{0, 1, 3, 4, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_5D,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder5D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes5D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

/**
 * 6D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes6D = {
        {{2, 8, 5, 13, 11, 16}},
        {{2, 11, 6, 2, 15, 10}},
        {{2, 13, 1, 3, 14, 32}},
        {{2, 14, 3, 4, 4, 22}},
};

const std::vector<std::vector<size_t>> inputOrder6D = {
        // use permute_ref
        std::vector<size_t>{0, 4, 3, 5, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 4, 5, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_6D,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder6D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

/**
 * 8D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes8D = {
        {{1, 2, 3, 4, 5, 6, 7, 8}},
};

const std::vector<std::vector<size_t>> inputOrder8D = {
        std::vector<size_t>{1, 2, 4, 3, 6, 7, 5, 0},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_8D,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder8D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes8D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

}  // namespace

namespace ov {
namespace test {

transposeParams make7DTransposeParams();
transposeParams make8DTransposeParams();

class TransposeInferTwiceBaseTest : public TransposeLayerTest {
protected:
    void run_infer_twice() {
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        generate_inputs(targetStaticShapes[0]);
        for (const auto& input : inputs) {
            infer_request.set_tensor(input.first, input.second);
        }
        infer_request.infer();
        auto actual_output1 = infer_request.get_output_tensor(0);
        auto expected_outputs1 = calculate_refs();
        ASSERT_EQ(expected_outputs1.size(), 1);
        compare(expected_outputs1, {actual_output1});

        inputs.clear();
        generate_inputs(targetStaticShapes[1]);
        for (const auto& input : inputs) {
            infer_request.set_tensor(input.first, input.second);
        }
        infer_request.infer();
        auto actual_output2 = infer_request.get_output_tensor(0);
        auto expected_outputs2 = calculate_refs();
        ASSERT_EQ(expected_outputs2.size(), 1);
        compare(expected_outputs2, {actual_output2});
    }
};

transposeParams make8DTransposeParams() {
    std::vector<size_t> input_order = {7, 6, 5, 4, 3, 2, 1, 0};
    ov::element::Type model_type = ov::element::f32;
    ov::PartialShape dynamic_shape = ov::PartialShape::dynamic(8);
    std::vector<ov::Shape> static_shapes = {{2, 3, 4, 5, 6, 7, 8, 9}, {2, 2, 4, 6, 6, 7, 2, 8}};
    std::vector<InputShape> input_shapes = {{dynamic_shape, static_shapes}};
    std::string target_device = ov::test::utils::DEVICE_GPU;
    return {input_order, model_type, input_shapes, target_device};
}

class Transpose8DInferTwiceTest : public TransposeInferTwiceBaseTest {};

TEST_P(Transpose8DInferTwiceTest, infer_twice_diff_shapes_same_request) {
    run_infer_twice();
}

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_8D_Infer_Twice, Transpose8DInferTwiceTest, ::testing::Values(make8DTransposeParams()));

transposeParams make7DTransposeParams() {
    std::vector<size_t> input_order = {6, 5, 4, 3, 2, 1, 0};
    ov::element::Type model_type = ov::element::f32;
    ov::PartialShape dynamic_shape = ov::PartialShape::dynamic(7);
    std::vector<ov::Shape> static_shapes = {{2, 3, 4, 5, 6, 7, 8}, {2, 3, 4, 5, 6, 8, 7}};
    std::vector<InputShape> input_shapes = {{dynamic_shape, static_shapes}};
    std::string target_device = ov::test::utils::DEVICE_GPU;
    return {input_order, model_type, input_shapes, target_device};
}

class Transpose7DInferTwiceTest : public TransposeInferTwiceBaseTest {};

TEST_P(Transpose7DInferTwiceTest, infer_twice_diff_shapes_same_request) {
    run_infer_twice();
}

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_7D_Infer_Twice, Transpose7DInferTwiceTest, ::testing::Values(make7DTransposeParams()));
}  // namespace test
}  // namespace ov