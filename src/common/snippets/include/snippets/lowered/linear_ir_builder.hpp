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
        Config(bool deep_copy_of_shapes_ = true) : deep_copy_of_shapes(deep_copy_of_shapes_) {}

        // If True, copy of stored pointer in `PortDescriptor::m_tensor_shape`.
        // If False, copy shapes as shared pointers.
        const bool deep_copy_of_shapes = true;
    };

    LinearIRBuilder(Config config = {}) : m_config(std::move(config)) {}

    /**
     * @brief Make a full copy of LinearIR by rules described in `m_config`
     * @param linear_ir Linear IR
     * @param expression_map expression map
     * @return clone of `linear_ir`
     */
    inline std::shared_ptr<LinearIR> clone(const std::shared_ptr<LinearIR>& linear_ir,  ExpressionMap& expression_map) const {
        auto result = std::make_shared<LinearIR>();
        clone(linear_ir.get(), result.get(), expression_map);
        return result;
    }
    inline std::shared_ptr<LinearIR> clone(const std::shared_ptr<LinearIR>& linear_ir) const {
        ExpressionMap expression_map;
        return clone(linear_ir, expression_map);
    }
    inline LinearIR clone(const LinearIR& linear_ir) const {
        LinearIR result;
        ExpressionMap expression_map;
        clone(&linear_ir, &result, expression_map);
        return result;
    }
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
    void clone(const LinearIR* src, LinearIR* dst,  ExpressionMap& expression_map) const;
    Config m_config = {};
};

} // namespace lowered
} // namespace snippets
} // namespace ov
