// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_config.hpp"

#include <utility>

#include "cpu_types.h"
#include "nodes/executors/executor.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

legacy::MVNExecutor::MVNExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

VectorDims legacy::MVNExecutor::transformTo5DCase(const VectorDims& shape, bool initAcrossChannels) {
    switch (shape.size()) {
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under the unified 5d procedure.
    // otherwise there are not enough data in spatial dimension to process in one kernel.
    case 1:  // C
        if (initAcrossChannels) {
            return VectorDims({1, 1, 1, 1, shape[0]});
        } else {
            return VectorDims({1, shape[0], 1, 1, 1});
        }
    case 2:  // NC
        if (initAcrossChannels) {
            return VectorDims({1, shape[0], 1, shape[1], 1});
        } else {
            return VectorDims({shape[0], shape[1], 1, 1, 1});
        }
    case 3: {
        return VectorDims({shape[0], shape[1], 1, shape[2], 1});
    }
    case 4: {
        return VectorDims({shape[0], shape[1], 1, shape[2], shape[3]});
    }
    case 5: {
        return VectorDims({shape[0], shape[1], shape[2], shape[3], shape[4]});
    }
    default: {
        OPENVINO_THROW("MVN executor doesn't support planar layout with rank: ", shape.size());
    }
    }
}

legacy::MVNExecutorBase::MVNExecutorBase(const MVNAttrs& mvnAttrs)
    : mvnAttrs(mvnAttrs),
      src_data_size(mvnAttrs.src_prc.size()),
      dst_data_size(mvnAttrs.dst_prc.size()) {}

}  // namespace ov::intel_cpu