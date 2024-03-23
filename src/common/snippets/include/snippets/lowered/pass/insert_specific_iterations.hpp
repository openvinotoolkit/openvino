// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/runtime_config.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/loop.hpp"

#include <array>


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertSpecificIterations
 * @brief Inserts separate loop bodies for first/last iterations if needed.
 * Also calls previously registered SpecificIterationHandlers for the inserted bodies and the main body.
 * @ingroup snippets
 */
class InsertSpecificIterations : public RangedPass {
public:
    OPENVINO_RTTI("InsertSpecificIterations", "RangedPass")
    InsertSpecificIterations() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    /**
     * @brief Makes a copy of a loop body with id 'loop_id' and inserts it to the LinearIR before the 'insert_pos' position
     * @param linear_ir LinearIR which should be modified
     * @param loop_id id of the loop which should be copied
     * @param insert_pos position before which the loop body copy should be inserted
     * @return iterator which points on the LoopBegin copy
     */
    static LinearIR::constExprIt insert_copy_loop(LinearIR& linear_ir, const size_t loop_id, const LinearIR::constExprIt& insert_pos);
    /**
     * @brief Makes a copy of a loop body if needed and inserted before the existing loop (if there are other non-inserted specific iterations),
     *        initializes loop parameters from the corresponding descriptor and apply handlers
     * @param linear_ir LinearIR which should be modified
     * @param begin iterator of LoopBegin
     * @param end iterator of LoopEnd
     * @param loop_info LoopInfo of the loop which should be created
     * @param loop_id id of the loop which should be created
     * @param runtime_config config that contains all loop descriptors
     * @param loop_end LoopEnd of the loop
     * @param type type of the specific iterations
     * @return true if the loop has been created, otherwise returns false
     */
    bool create_specific_loop(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                              const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id, const RuntimeConfig& runtime_config,
                              const std::shared_ptr<op::LoopEnd>& loop_end, const RuntimeConfig::LoopDescriptor::Type& type);
    /**
     * @brief Initializes loop parameters from the corresponding descriptor and apply handlers
     * @param loop_end LoopEnd of the loop
     * @param desc descriptor that contains all needed parameters of the loop
     * @param linear_ir LinearIR which should be modified
     * @param handlers sets of passes that will be applied on this sup-loop
     * @param begin iterator of LoopBegin
     * @param end iterator of LoopEnd
     */
    void init_specific_loop(const std::shared_ptr<op::LoopEnd>& loop_end, const RuntimeConfig::LoopDescriptor& desc,
                            LinearIR& linear_ir, const PassPipeline& handlers, LinearIR::constExprIt begin, LinearIR::constExprIt end);

    /**
     * @brief Returns specific ietartions handlers by loop descriptor type
     * @param loop_info LoopInfo of the loop which should be updated
     * @param type type of the specific iterations
     * @return sets of passes that will be applied on this sup-loop
     */
    static PassPipeline get_iter_specific_handlers_by_type(const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                                           const RuntimeConfig::LoopDescriptor::Type& type);

    static std::array<RuntimeConfig::LoopDescriptor::Type, 3> m_loop_types;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
