// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/factory.hpp"

#include <mutex>

#include "openvino/core/node.hpp"

namespace ov {
#ifndef _WIN32
template class FactoryRegistry<Node>;
#endif
}  // namespace ov
