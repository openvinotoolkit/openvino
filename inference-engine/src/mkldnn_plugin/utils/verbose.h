// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include "mkldnn_node.h"

namespace MKLDNNPlugin {

void print(const MKLDNNNodePtr& node, const std::string& verboseLvl);

} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
