// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "port_descriptor.hpp"

#include "expression_port.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

class Expression;

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(ExpressionPort source_descriptor, const std::set<ExpressionPort>& consumer_descriptors = {});

    const ExpressionPort& get_source() const { return m_source_port; }
    std::set<ExpressionPort> get_consumers() const { return m_consumer_ports; }

    void add_consumer(const ExpressionPort& consumer);
    void remove_consumer(const ExpressionPort& consumer);
    bool found_consumer(const ExpressionPort& consumer) const;
    std::set<ExpressionPort>::const_iterator find_consumer(const ExpressionPort& consumer) const;
    std::set<ExpressionPort>::iterator find_consumer(const ExpressionPort& consumer);

private:
    ExpressionPort m_source_port;
    std::set<ExpressionPort> m_consumer_ports;
};
using TensorPtr = std::shared_ptr<Tensor>;


} // namespace lowered
} // namespace snippets
} // namespace ngraph
