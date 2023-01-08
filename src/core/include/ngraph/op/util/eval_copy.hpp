// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define COPY_TENSOR(a)       \
    case element::Type_t::a: \
        rc = copy_tensor<element::Type_t::a>
