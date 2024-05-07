// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"

namespace ov {
namespace snippets {
namespace lowered {


/* The helper class that can make a copy of LinearIR or range of it by specific rules.
 * The rules are described in config.
 */
class LinearIRBuilder {
public:
    struct Config {
        Config(bool deep_copy_of_shapes_ = true, bool copy_missed_consumers_ = true)
            : deep_copy_of_shapes(deep_copy_of_shapes_), copy_missed_consumers(copy_missed_consumers_) {}

        // If True, copy of stored pointer in `PortDescriptor::m_tensor_shape`.
        // If False, copy shapes as shared pointers.
        const bool deep_copy_of_shapes = true;
        // At the moment, input port of expression must have only one source.
        // However, for example, after LinearIR range insertion to the LinearIR (InsertSpecificIteration pass)
        // several operations can write to the same consumer: several `Store` ops from different loop bodies store to the same Buffer/Result.
        // Since `clone` algorithm is linear and during expression cloning creates only input port connectors from sources,
        // algorithm can miss some consumers. For example:
        //      The consumers of Store0 : Buffer0
        //      The consumers of Store1 : Buffer0
        // The result: Buffer0 has only one source in input connector - Store1
        // Algorithm automatically doesn't add Buffer to consumers of Store0. Thus,
        // If True, `clone` algorithm add missed consumers.
        // If False, cloned LinearIR will be built by default (without extra consumers).
        const bool copy_missed_consumers = true;
    };

    LinearIRBuilder(Config config = {}) : m_config(std::move(config)) {}

    /**
     * @brief Make a full copy of LinearIR by rules described in `m_config`
     * @param linear_ir Linear IR
     * @return clone of `linear_ir`
     */
    std::shared_ptr<LinearIR> clone(const std::shared_ptr<LinearIR>& linear_ir) const;
    /**
     * @brief Make a copy of LinearIR range by rules described in `m_config`
     * @param begin begin iterator of the target range of LinearIR
     * @param end end iterator of the target range of LinearIR
     * @param expression_map expression map
     * @return cloned range of `linear_ir`
     */
    LinearIR::container clone_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end,
                                    ExpressionMap& expression_map) const;

private:
    Config m_config = {};
};

} // namespace lowered
} // namespace snippets
} // namespace ov
