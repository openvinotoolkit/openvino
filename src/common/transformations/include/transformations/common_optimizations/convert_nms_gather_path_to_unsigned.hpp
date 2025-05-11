// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNmsGatherPathToUnsigned;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Converts Gather indices to unsigned if indices are from NMS selected indices output.
 * NMS returns -1 for not selected boxes, old version of Gather fill corresponding
 * output for such indices with zero.
 * But new Gather-8 has support of negative indices indicating counting from the end.
 * In order to keep such behaviour (until dynamism is not supported) instead of -1 new
 * Gather-8 will accept UINT32_MAX which is always outside of the bounds
 * and corresponding output for such indices in gather always will be filled with zeros.
 */
class ov::pass::ConvertNmsGatherPathToUnsigned : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertNmsGatherPathToUnsigned");
    ConvertNmsGatherPathToUnsigned();
};
