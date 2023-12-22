// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/factory.hpp"

#include <mutex>

#include "ngraph/node.hpp"

using namespace std;

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph {
mutex& get_registry_mutex() {
    static mutex registry_mutex;
    return registry_mutex;
}

#ifndef _WIN32
template class FactoryRegistry<ngraph::Node>;
#endif
}  // namespace ngraph
