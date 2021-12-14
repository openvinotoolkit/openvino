// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_priorgridgenerator_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsExpPriorGridGeneratorAllInputs_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ExpPriorGridGeneratorLayerInputs),
        ::testing::ValuesIn(s_ExpPriorGridGeneratorLayerParam))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsExpPriorGridGeneratorNoFeatureMap_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ExpPriorGridGenLayerNoFMInputs),
        ::testing::ValuesIn(s_ExpPriorGridGenLayerNoFMParam))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsExpPriorGridGeneratorNoInputImage_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ExpPriorGridGenLayerNoInputImage),
        ::testing::ValuesIn(s_ExpPriorGridGenLayerNoInputImageParam))
);
