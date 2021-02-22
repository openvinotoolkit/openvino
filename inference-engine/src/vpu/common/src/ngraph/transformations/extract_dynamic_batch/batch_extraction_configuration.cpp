// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/error.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/batch_extraction_configuration.hpp"

namespace vpu {

SliceConfiguration::SliceConfiguration(std::vector<SliceMode> inputs, std::vector<SliceMode> outputs)
   : m_isSliceSupported(true)
   , m_inputs(std::move(inputs))
   , m_outputs(std::move(outputs)) {}

bool SliceConfiguration::isSliceSupported() const {
    return m_isSliceSupported;
}

const std::vector<SliceMode>& SliceConfiguration::inputs() const {
    VPU_THROW_UNLESS(m_isSliceSupported, "Encountered an attempt to access inputs slice configuration for a case when slice is unsupported");
    return m_inputs;
}

const std::vector<SliceMode>& SliceConfiguration::outputs() const {
    VPU_THROW_UNLESS(m_isSliceSupported, "Encountered an attempt to access outputs slice configuration for a case when slice is unsupported");
    return m_outputs;
}

}  // namespace vpu

