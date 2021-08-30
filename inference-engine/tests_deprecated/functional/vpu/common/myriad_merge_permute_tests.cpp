// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_merge_permute_tests.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy_3D, myriadLayersMergePermuteNDTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_3D)
          , ::testing::ValuesIn(s_permuteParams_3D)
));

INSTANTIATE_TEST_SUITE_P(accuracy_4D, myriadLayersMergePermuteNDTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_4D)
          , ::testing::ValuesIn(s_permuteParams_4D)
));

INSTANTIATE_TEST_SUITE_P(accuracy_5D, myriadLayersMergePermuteNDTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_5D)
          , ::testing::ValuesIn(s_permuteParams_5D)
));
