// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_priorgridgenerator_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsExpPriorGridGenerator_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ExpPriorGridGeneratorLayerInputs),
        ::testing::ValuesIn(s_ExpPriorGridGeneratorLayerParam))
);
