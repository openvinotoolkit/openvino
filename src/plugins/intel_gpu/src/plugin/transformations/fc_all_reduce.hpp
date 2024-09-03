// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
/**
 * @ingroup ov_transformation_common_api
 * @brief FCALLReduce transformation matches following graph:
*/
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace ov {
namespace intel_gpu {

class FCALLReduce: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FCALLReduce", "0");
    FCALLReduce(size_t world_size, size_t world_rank);
};

}   // namespace intel_gpu
}   // namespace ov
