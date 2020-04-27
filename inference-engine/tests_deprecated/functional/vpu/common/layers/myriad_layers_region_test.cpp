// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_region_test.hpp"

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayerRegionYolo_nightly,
        ::testing::ValuesIn(s_regionData)
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsRegion_CHW_HW_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 125, 13, 13})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(125)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsRegion_CHW_HW_80cl_nightly,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 425, 13, 13})
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(425)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerRegionYolo_CHW_nightly,
        ::testing::ValuesIn(s_classes)
);
