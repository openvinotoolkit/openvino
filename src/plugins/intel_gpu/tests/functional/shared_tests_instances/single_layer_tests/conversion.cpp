// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "conversion.hpp"

namespace ov {
namespace test {

void ConvertToBooleanLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto shape = targetInputStaticShapes.front();
        auto size = shape_size(shape);
        auto input_type = funcInputs[0].get_element_type();

        ov::Tensor tensor = ov::Tensor(input_type, shape);
        const auto first_part_size = size / 2;
        const auto second_part_size = size - first_part_size;

        // 1). Validate the nearest to zero values (Abs + Ceil)
        {
                double start_from = -2;
                uint32_t range = 4;
                int32_t resolution = size;
                if (input_type == ov::element::f32) {
                auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
                ov::test::utils::fill_data_random(rawBlobDataPtr, first_part_size, range, start_from, resolution);
                } else if (input_type == ov::element::f16) {
                auto* rawBlobDataPtr = static_cast<ov::float16*>(tensor.data());
                ov::test::utils::fill_data_random(rawBlobDataPtr, first_part_size, range, start_from, resolution);
                } else {
                FAIL() << "Generating inputs with precision " << input_type.to_string() << " isn't supported, if output precision is boolean.";
                }
        }

        // 2). Validate the values that are more than UINT8_MAX in absolute (Abs + Min)
        {
                ov::test::utils::InputGenerateData in_data_neg;
                double neg_start_from = -1.5 * std::numeric_limits<uint8_t>::max();
                double pos_start_from = 0.5 * std::numeric_limits<uint8_t>::max();
                uint32_t range = 256;
                auto neg_size = second_part_size / 2;
                auto pos_size = second_part_size - neg_size;
                int32_t resolution = 1;

                if (input_type == ov::element::f32) {
                auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
                ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size, neg_size, range, neg_start_from, resolution);
                ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size + neg_size, pos_size, range, pos_start_from, resolution);
                } else if (input_type == ov::element::f16) {
                auto* rawBlobDataPtr = static_cast<ov::float16*>(tensor.data());
                ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size, neg_size, range, neg_start_from, resolution);
                ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size + neg_size, pos_size, range, pos_start_from, resolution);
                } else {
                FAIL() << "Generating inputs with precision " << input_type.to_string() << " isn't supported, if output precision is boolean.";
                }
        }

        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
}

}  // namespace test
}  // namespace ov


namespace {
using ov::test::ConversionLayerTest;
using ov::test::ConvertToBooleanLayerTest;

const std::vector<ov::test::utils::ConversionTypes> conversionOpTypes = {
    ov::test::utils::ConversionTypes::CONVERT,
    ov::test::utils::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<ov::Shape>> inShape = {{{1, 2, 3, 4}}};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
        ov::element::i8,
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(conversionOpTypes),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConversionLayerTest::getTestCaseName);

TEST_P(ConvertToBooleanLayerTest, CompareWithRefs) {
    run();
};

const std::vector<ov::element::Type> precisions_floating_point = {
        ov::element::f32,
        ov::element::f16
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConvertToBooleanLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn({ov::test::utils::ConversionTypes::CONVERT}),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ov::element::boolean),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvertToBooleanLayerTest::getTestCaseName);

}  // namespace
