// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mutex>

#include "ngraph/factory.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"

using namespace std;

namespace ngraph
{
    mutex& get_registry_mutex()
    {
        static mutex registry_mutex;
        return registry_mutex;
    }
} // namespace ngraph
