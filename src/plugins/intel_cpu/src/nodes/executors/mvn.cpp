// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;

MVNExecutor::MVNExecutor(const ExecutorContext::CPtr context) : context(context) {}

SizeVector MVNExecutor::transformTo5DCase(const SizeVector& shape, bool initAcrossChannels) {
    switch (shape.size()) {
        // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under unified 5d procedure.
        // otherwise there are not enough data in spatial dimension to process in one kernel.
        case 1 :  // C
            if (initAcrossChannels) {
                return SizeVector({1, 1, 1, 1, shape[0]});
            } else {
                return SizeVector({1, shape[0], 1, 1, 1});
            }
        case 2 :  // NC
            if (initAcrossChannels) {
                return SizeVector({1, shape[0], 1, shape[1], 1});
            } else {
                return SizeVector({shape[0], shape[1], 1, 1, 1});
            }
        case 3 : { return SizeVector({shape[0], shape[1], 1, shape[2], 1}); }
        case 4 : { return SizeVector({shape[0], shape[1], 1, shape[2], shape[3]}); }
        case 5 : { return SizeVector({shape[0], shape[1], shape[2], shape[3], shape[4]}); }
        default : { OPENVINO_THROW("MVN executor doesn't support planar layout with rank: ", shape.size()); }
    }
}

}   // namespace intel_cpu
}   // namespace ov