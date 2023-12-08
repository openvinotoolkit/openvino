// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/op/loop.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertLoops
 * @brief The pass explicitly insert LoadBegin and LoadEnd in Linear IR using LoopManager::LoopInfo from Loop markup algorithm
 * @ingroup snippets
 */
class InsertLoops : public Pass {
public:
    OPENVINO_RTTI("InsertLoops", "Pass")
    InsertLoops() = default;
    bool run(LinearIR& linear_ir) override;
private:
    static void insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id);
    static void filter_ports(std::vector<LinearIR::LoopManager::LoopPort>& loop_entries, std::vector<LinearIR::LoopManager::LoopPort>& loop_exits);
    static std::shared_ptr<op::LoopBegin> make_loop_begin(bool is_dynamic);
    static std::shared_ptr<op::LoopEnd> make_loop_end(bool is_dynamic, const Output<Node>& loop_begin, size_t work_amount, size_t work_amount_increment,
                                                      std::vector<bool> is_incremented, std::vector<int64_t> ptr_increments,
                                                      std::vector<int64_t> finalization_offsets, std::vector<int64_t> element_type_sizes,
                                                      size_t input_num, size_t output_num, size_t id);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
