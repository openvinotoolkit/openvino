// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dims_tests.hpp"

PLUGING_CASE_WITH_SUFFIX(GPU, _smoke, IO_BlobTest, params);

#if defined(ENABLE_MKL_DNN)
    PLUGING_CASE(HETERO, IO_BlobTest, params);
#endif
