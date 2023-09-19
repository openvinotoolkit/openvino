// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeinfo>

#include "snippets/pass/common_optimizations.hpp"


namespace ov {
namespace snippets {
namespace pass {

/**
 * @brief Base class for Subgraph passes
 * @ingroup snippets
 */
class CommonOptimizations::SubgraphPass {
public:
    SubgraphPass() = default;
    virtual ~SubgraphPass() = default;

    virtual bool run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) = 0;

    void set_name(const std::string& name) { m_name = name; }
    std::string get_name() const { return m_name; }

    using type_info_t = DiscreteTypeInfo;
    virtual const type_info_t& get_type_info() const = 0;

private:
    std::string m_name;
};


} // namespace pass
} // namespace snippets
} // namespace ov
