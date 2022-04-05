// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief ExperimentalDetectronTopKROIs-6 primitive
/// @details
struct experimental_detectron_topk_rois : public primitive_base<experimental_detectron_topk_rois> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_topk_rois)

    /**
     * Construct ExperimentalDetectronTopKROIs privitive.
     * @param id primitive id
     * @param inputs inputs parameters ids
     * @param max_rois maximal numbers of output ROIs.
     */
    experimental_detectron_topk_rois(const primitive_id &id, const std::vector<primitive_id> &inputs,
                                     const size_t max_rois,
                                     const padding &output_padding = padding())
            : primitive_base(id, inputs, "", output_padding),
              max_rois(max_rois) {}

    /// maximal numbers of output ROIs.
    size_t max_rois;
};

}  // namespace cldnn
