// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/topological_order.hpp"

#include "ngraph/node.hpp"

namespace ngraph {
void Order::reset(Order::Ptr order) {
    auto e = m_begin;
    int64_t cnt{0};
    while (e) {
        ++cnt;
        e->node->m_order = order;
        e->reset_id();
        if (e == m_end) {
            break;
        }
        e = e->output;
    }
    assert(cnt == m_size);
    m_size = 0;
}

void Order::remove(OrderElement::Ptr element) {
    auto output = element->output;
    auto input = element->input;
    if (output) {
        output->input = input;
    }
    if (input) {
        input->output = output;
    }
    if (element == m_begin) {
        m_begin = output;
    }
    if (element == m_end) {
        m_end = input;
    }
    --m_size;
}
}  // namespace ngraph
