// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/node_registry.hpp"

void ov::pass::NodeRegistry::clear() {
    m_nodes.clear();
}
