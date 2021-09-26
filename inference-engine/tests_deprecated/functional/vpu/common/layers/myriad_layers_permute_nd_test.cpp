// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_permute_nd_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy_2D, myriadLayersPermuteNDTests_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_2D)
          , ::testing::ValuesIn(s_permuteTensors_2D)
          , ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
          , ::testing::ValuesIn(s_permutePrecisions)
));

INSTANTIATE_TEST_SUITE_P(accuracy_3D, myriadLayersPermuteNDTests_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_3D)
          , ::testing::ValuesIn(s_permuteTensors_3D)
          , ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
          , ::testing::ValuesIn(s_permutePrecisions)
));

INSTANTIATE_TEST_SUITE_P(accuracy_4D, myriadLayersPermuteNDTests_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_4D)
          , ::testing::ValuesIn(s_permuteTensors_4D)
          , ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
          , ::testing::ValuesIn(s_permutePrecisions)
));

INSTANTIATE_TEST_SUITE_P(accuracy_5D, myriadLayersPermuteNDTests_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_5D)
          , ::testing::ValuesIn(s_permuteTensors_5D)
          , ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
          , ::testing::ValuesIn(s_permutePrecisions)
));

INSTANTIATE_TEST_SUITE_P(fc_to_conv_case, myriadLayersPermuteNDTests_smoke,
    ::testing::Values(
        std::make_tuple(
            SizeVector{8, 50, 256, 7, 7},
            SizeVector{2, 0, 3, 1, 4},
            IRVersion::v7,
            InferenceEngine::Precision::FP16
        ),
        std::make_tuple(
            SizeVector{1024, 8, 1, 50, 1},
            SizeVector{1, 3, 0, 2, 4},
            IRVersion::v7,
            InferenceEngine::Precision::FP16
        )
    )
);

INSTANTIATE_TEST_SUITE_P(accuracy_FasterRCNN, myriadLayersPermuteNDTests_smoke,
        ::testing::Combine(
             ::testing::Values<InferenceEngine::SizeVector>({1, 24, 14, 14})
            ,::testing::Values<InferenceEngine::SizeVector>({0, 2, 3, 1})
            ,::testing::Values<IRVersion>(IRVersion::v7)
            ,::testing::ValuesIn(s_permutePrecisions)
            ));

INSTANTIATE_TEST_SUITE_P(accuracy_MaskRCNN, myriadLayersPermuteNDTests_smoke,
        ::testing::Combine(
             ::testing::Values<InferenceEngine::SizeVector>({4, 3, 1, 88, 120})
            ,::testing::Values<InferenceEngine::SizeVector>({0, 3, 4, 1, 2})
            ,::testing::Values<IRVersion>(IRVersion::v7)
            ,::testing::ValuesIn(s_permutePrecisions)
            ));
