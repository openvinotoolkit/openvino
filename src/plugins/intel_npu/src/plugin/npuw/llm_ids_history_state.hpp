// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

class LLMIdsHistoryState : public ov::IVariableState {
public:
    using ov::IVariableState::IVariableState;

    LLMIdsHistoryState(const std::string& name, const std::size_t seq_len_dim)
        : ov::IVariableState(name),
          m_seq_len_dim(seq_len_dim) {
        std::vector<uint64_t> dims(4, 1);
        dims[m_seq_len_dim] = 0;
        m_ids_history = ov::make_tensor(ov::element::i64, ov::Shape{dims});
        ov::IVariableState::set_state(m_ids_history);
    }

    void reset() override final {
        std::vector<uint64_t> dims(4, 1);
        dims[m_seq_len_dim] = 0;
        m_ids_history->set_shape(ov::Shape{dims});
        ov::IVariableState::set_state(m_ids_history);
    }

protected:
    uint64_t m_seq_len_dim{0u};
    ov::SoPtr<ov::ITensor> m_ids_history;
};
