// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repacked_input.hpp"

namespace ov::intel_cpu {

RepackedInput::RepackedInput(std::shared_ptr<const RepackedInputKernel> kernel,
                             CpuBlockedMemoryDescPtr desc,
                             VectorDims in_offsets,
                             VectorDims out_offsets)
    : m_kernel(std::move(kernel)),
      m_desc(std::move(desc)),
      m_in_offsets(std::move(in_offsets)),
      m_out_offsets(std::move(out_offsets)) {
    OPENVINO_ASSERT(m_in_offsets.size() == m_out_offsets.size(), "Incorrect size of offsets");
    OPENVINO_ASSERT(m_desc, "Descriptor is empty");
}

const CpuBlockedMemoryDescPtr& RepackedInput::desc() const {
    return m_desc;
}

const VectorDims& RepackedInput::in_offsets() const {
    return m_in_offsets;
}

const VectorDims& RepackedInput::out_offsets() const {
    return m_out_offsets;
}
}  // namespace ov::intel_cpu
