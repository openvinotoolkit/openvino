// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_interpolate.hpp"
#include <memory>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "ref_interpolate_legacy.hpp"

namespace ov::intel_cpu {

CommonInterpolateExecutor::CommonInterpolateExecutor(const InterpolateAttrs &attrs, const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr  /*context*/) {
    refExecutorLegacy = std::make_shared<legacy::InterpolateRefExecutorLegacy>(attrs,
                                                                               memory.at(0)->getDescPtr()->getShape().getDims(),
                                                                               memory.at(1)->getDescPtr()->getShape().getDims(),
                                                                               attrs.dataScales);
}

void CommonInterpolateExecutor::execute(const MemoryArgs &memory) {
    refExecutorLegacy->exec(static_cast<const uint8_t *>(memory.at(0)->getData()),
                            static_cast<uint8_t *>(memory.at(1)->getData()),
                            static_cast<const uint8_t *>(memory.at(2)->getData()));
}

bool CommonInterpolateExecutor::update(const MemoryArgs & /*memory*/) {
    return true;
}

bool CommonInterpolateExecutor::supports(const InterpolateConfig&  /*config*/) {
    return true;
}


}  // namespace ov::intel_cpu
