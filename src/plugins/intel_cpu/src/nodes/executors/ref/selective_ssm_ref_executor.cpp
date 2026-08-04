// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/ref/selective_ssm_ref_executor.hpp"

#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/ref/selective_ssm_ref_kernel.hpp"

namespace ov::intel_cpu {

namespace {

bool is_supported_precision(const ov::element::Type& precision) {
    return precision == ov::element::f32 || precision == ov::element::f16 || precision == ov::element::bf16;
}

}  // namespace

bool SelectiveSSMRefExecutor::supports(const SelectiveSSMConfig& config) {
    return is_supported_precision(config.descs.at(ARG_SSM_A)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_DT)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_B)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_X)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_C)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_STATE)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_OUT)->getPrecision()) &&
           is_supported_precision(config.descs.at(ARG_SSM_OUT_STATE)->getPrecision());
}

SelectiveSSMRefExecutor::SelectiveSSMRefExecutor(const SelectiveSSMAttrs& attrs,
                                                 const MemoryArgs& memory,
                                                 [[maybe_unused]] ExecutorContext::CPtr context)
    : m_attrs(attrs) {
    update(memory);
}

bool SelectiveSSMRefExecutor::update([[maybe_unused]] const MemoryArgs& memory) {
    return true;
}

void SelectiveSSMRefExecutor::execute(const MemoryArgs& memory) {
    PlainTensor A(memory.at(ARG_SSM_A));
    PlainTensor dt(memory.at(ARG_SSM_DT));
    PlainTensor B(memory.at(ARG_SSM_B));
    PlainTensor x(memory.at(ARG_SSM_X));
    PlainTensor C(memory.at(ARG_SSM_C));
    PlainTensor recurrent_state(memory.at(ARG_SSM_STATE));
    PlainTensor output(memory.at(ARG_SSM_OUT));
    PlainTensor output_recurrent_state(memory.at(ARG_SSM_OUT_STATE));

    detail::selective_ssm_reference(A, dt, B, x, C, recurrent_state, output, output_recurrent_state);
}

impl_desc_type SelectiveSSMRefExecutor::implType() const {
    return impl_desc_type::ref_any;
}

}  // namespace ov::intel_cpu
