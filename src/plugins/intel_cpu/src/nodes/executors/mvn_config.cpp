// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_config.hpp"

namespace ov {
namespace intel_cpu {

VectorDims transformTo5DCase(const VectorDims& shape, MVNAttrs& mvnAttrs) {
    VectorDims shape5D;
    size_t rank = shape.size();
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under unified 5d procedure.
    // otherwise there are not enough data in spatial dimension to process in one kernel.
    switch (rank) {
        case 1 :  // C
            if (mvnAttrs.initAcrossChannels_) {
                shape5D = {1, 1, 1, 1, shape[0]};
                mvnAttrs.execAcrossChannels_ = false;
                break;
            } else {
                shape5D = {1, shape[0], 1, 1, 1};
                break;
            }
        case 2 :  // NC
            if (mvnAttrs.initAcrossChannels_) {
                shape5D = {1, shape[0], 1, shape[1], 1};
                mvnAttrs.execAcrossChannels_ = false;
                break;
            } else {
                shape5D = {shape[0], shape[1], 1, 1, 1};
                break;
            }
        case 3 : { shape5D = {shape[0], shape[1], 1, shape[2], 1}; break; }
        case 4 : { shape5D = {shape[0], shape[1], 1, shape[2], shape[3]}; break; }
        case 5 : { shape5D = {shape[0], shape[1], shape[2], shape[3], shape[4]}; break; }
        default: {
            OPENVINO_THROW("MVN layer with name doesn't support planar layout with rank: ", shape.size());
        }
    }
    return shape5D;
}

}   // namespace intel_cpu
}   // namespace ov