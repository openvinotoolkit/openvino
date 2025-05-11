// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/recurrent_cell_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true)
};

namespace testValues1 {

const std::vector<LayerTestsDefinitions::RecurrentCellTransformationParam> params = {
    // LSTMSequence
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        ov::builder::subgraph::RecurrentCellFunction::RNNType::LSTMSequence,
        "RNNCell",
        "U8"
    },
    // asymmetrical FQ on weights
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // W
        {256ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        // R
        {256ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        ov::builder::subgraph::RecurrentCellFunction::RNNType::LSTMSequence,
        "RNNCell",
        "FP32"
    }
};

const std::vector<std::vector<ov::PartialShape>> activations_shapes = {{{1, 1, 16}, {1, 1, 128}, {1, 1, 128}}};
const std::vector<std::vector<ov::Shape>> weights_shapes = {{{1, 512, 16}, {1, 512, 128}, {1, 512}}};

// Quantized Recurrent models are not supported by GPU yet. Keep tests for future
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_LPT, RecurrentCellTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn({ true, false }),
        ::testing::ValuesIn(params)),
    RecurrentCellTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {

const std::vector<LayerTestsDefinitions::RecurrentCellTransformationParam> params = {
    // GRUSequence
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        ov::builder::subgraph::RecurrentCellFunction::RNNType::GRUSequence,
        "RNNCell",
        "U8"
    },
    // asymmetrical FQ on weights
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {ov::element::f32},
             {},
             {0.01f},
        },
        // W
        {256ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        // R
        {256ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        ov::builder::subgraph::RecurrentCellFunction::RNNType::GRUSequence,
        "RNNCell",
        "FP32"
    }
};

const std::vector<std::vector<ov::PartialShape>> activations_shapes = {{{1, 2, 3}, {1, 2, 3}, {}}};
const std::vector<std::vector<ov::Shape>> weights_shapes = {{{1, 9, 3}, {1, 9, 3}, {1, 9}}};

// Quantized Recurrent models are not supported by GPU yet. Keep tests for future
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_LPT, RecurrentCellTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn({ true, false }),
        ::testing::ValuesIn(params)),
    RecurrentCellTransformation::getTestCaseName);
} // namespace testValues2
