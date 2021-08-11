// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>

#include "ngraph/env_util.hpp"

#include "ngraph/ngraph_visibility.hpp"

#include "ngraph/ngraph_namespace.hpp"

namespace ov
{
    NGRAPH_API
    void set_provenance_enabled(bool enabled);
    NGRAPH_API
    bool get_provenance_enabled();
} // namespace ov
