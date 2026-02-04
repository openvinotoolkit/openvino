// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface EliminateInplaceOps
 * @brief Eliminates operations that are effectively inplace (input == output).
 *        Currently handles Fill operations where offset equals register capacity,
 *        which means the operation doesn't actually fill any new data.
 *        This pass should run after InsertSpecificIterations and before InitRegisters.
 * @ingroup snippets
 */
class EliminateInplaceOps : public Pass {
public:
    OPENVINO_RTTI("EliminateInplaceOps", "", Pass);

    /**
     * @brief Callback type for determining if a Fill operation is inplace.
     *        Takes offset and element size, returns true if the Fill is inplace.
     */
    using IsInplaceFillCallback = std::function<bool(size_t offset, size_t element_size)>;

    /**
     * @brief Constructor with callback for inplace detection
     * @param is_inplace_fill_callback Function to determine if a Fill is inplace based on offset and element size
     */
    explicit EliminateInplaceOps(IsInplaceFillCallback is_inplace_fill_callback);

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass (true if any changes were made)
     */
    bool run(LinearIR& linear_ir) override;

private:
    /**
     * @brief Check if a Fill operation is inplace using the configured callback
     * @param fill_expr expression containing Fill operation
     * @return true if the Fill operation is inplace and can be eliminated
     */
    bool is_inplace_fill(const ExpressionPtr& fill_expr) const;

    /**
     * @brief Remove inplace Fill operation from the linear IR
     * @param linear_ir the target Linear IR
     * @param fill_expr expression containing inplace Fill operation
     */
    static void eliminate_fill(LinearIR& linear_ir, const ExpressionPtr& fill_expr);

    IsInplaceFillCallback m_is_inplace_fill_callback;
};

}  // namespace ov::snippets::lowered::pass
