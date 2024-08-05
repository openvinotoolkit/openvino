// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/kernel_executor_table.hpp"

namespace ov {
namespace snippets {

void KernelExecutorTable::replace_key_expression(const snippets::lowered::ExpressionPtr& from, const snippets::lowered::ExpressionPtr& to) {
    const auto& found = m_table.find(from);
    if (found != m_table.end()) {
        OPENVINO_ASSERT(m_table.count(to) == 0, "Attempt to replace a value that is already in the KernelExecutorTable");
        m_table.insert({to, found->second});
        m_table.erase(found);
    }
}

void KernelExecutorTable::reset_state(const ExecTableState& state) {
    OPENVINO_ASSERT(state.size() == m_table.size(), "Invalid state in restore_state: size mismatch");
    auto state_it = state.begin();
    for (const auto& table_record : m_table) {
        const auto& state_record = *state_it++;
        OPENVINO_ASSERT(table_record.first == state_record.first, "Invalid state in restore_state: expressions mismatch");
        table_record.second->update_by_config(*state_record.second);
    }
}

KernelExecutorTable::ExecTableState KernelExecutorTable::get_state() const {
    ExecTableState result;
    // Note: we need to clone configs when saving the state, since the configs still stored in the table can
    // be modified e.g. by calling update_by_expression();
    for (const auto& record : m_table)
        result.emplace_back(std::make_pair(record.first, record.second->get_config().get_clone_ptr()));
    return result;
}

}// namespace snippets
}// namespace ov
