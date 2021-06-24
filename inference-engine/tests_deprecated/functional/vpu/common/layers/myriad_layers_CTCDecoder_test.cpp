// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_CTCDecoder_test.hpp"

INSTANTIATE_TEST_SUITE_P(
	DISABLED_accuracy, myriadCTCDecoderLayerTests_smoke,
	::testing::Combine(
		::testing::Values<Dims>({{1, 88, 1, 71}}),
		::testing::Values<HwOptimization>(true, false),
		::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10),
		::testing::ValuesIn(s_CustomConfig)
));
