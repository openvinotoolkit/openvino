// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_grn_test.hpp"

INSTANTIATE_TEST_SUITE_P(
	accuracy, myriadLayersTestsGRN_smoke,
	::testing::Combine(
		::testing::ValuesIn(s_GRNInputs),
		::testing::Values<Bias>(0.5f, 10.f, 1.f),
		::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10),
		::testing::ValuesIn(s_CustomConfig)
));
