// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_net.hpp"

namespace CommonTestUtils {

void Port::connect(Port &port) {
    auto locked = m_parent.lock();
    if (!locked) {
        THROW_IE_EXCEPTION << "Error getting parent class";
    }
    auto &ir_net = locked->getNetwork();
    ir_net.addEdge(*this, port);
}

}  // namespace CommonTestUtils