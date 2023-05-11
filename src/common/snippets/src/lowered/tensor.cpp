// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/tensor.hpp"

#include <snippets/itt.hpp>
#include "snippets/utils.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

Tensor::Tensor(ExpressionPort source_descriptor, const std::set<ExpressionPort>& consumer_descriptors)
    : m_source_port(std::move(source_descriptor)), m_consumer_ports(consumer_descriptors) {}

std::set<ExpressionPort>::const_iterator Tensor::find_consumer(const ExpressionPort& consumer) const {
    // Note: Find by shared ptr and index port is enough since these parameters must be unique
    return std::find_if(m_consumer_ports.cbegin(), m_consumer_ports.cend(),
                        [&consumer](const ExpressionPort& td) {
                            return consumer.get_expr() == td.get_expr() && consumer.get_index() == td.get_index();
                        });
}

std::set<ExpressionPort>::iterator Tensor::find_consumer(const ExpressionPort& consumer) {
    // Note: Find by shared ptr and index port is enough since these parameters must be unique
    return std::find_if(m_consumer_ports.begin(), m_consumer_ports.end(),
                        [&consumer](const ExpressionPort& td) {
                            return consumer.get_expr() == td.get_expr() && consumer.get_index() == td.get_index();
                        });
}

bool Tensor::found_consumer(const ExpressionPort& consumer) const {
    return find_consumer(consumer) != m_consumer_ports.end();
}

void Tensor::add_consumer(const ExpressionPort& consumer) {
    OPENVINO_ASSERT(!found_consumer(consumer), "Consumer has been already added to Tensor!");
    const auto res = m_consumer_ports.insert(consumer);
    OPENVINO_ASSERT(res.second, "Consumer hasn't been added to the Tensor");
}

void Tensor::remove_consumer(const ExpressionPort& consumer) {
    const auto& found = find_consumer(consumer);
    OPENVINO_ASSERT(found != m_consumer_ports.end(), "Consumer is missed in Tensor!");
    m_consumer_ports.erase(found);
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
