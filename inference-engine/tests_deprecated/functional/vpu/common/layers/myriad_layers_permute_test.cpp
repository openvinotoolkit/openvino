// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_permute_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersPermuteTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors)
          , ::testing::ValuesIn(s_permuteTensors)
));

INSTANTIATE_TEST_CASE_P(accuracyFasterRCNN, myriadLayersPermuteTests_nightly,
        ::testing::Combine(
             ::testing::Values<InferenceEngine::SizeVector>({1, 24, 14, 14})
            ,::testing::Values<InferenceEngine::SizeVector>({0, 2, 3, 1})
            ));


