// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveRtInfo;

}  // namespace pass
}  // namespace ov

namespace ov {
namespace pass {
/**
 * @brief RemoveRtInfo transformation
 * @ingroup TODO ???
 */
class OPENVINO_API RemoveRtInfo : public ModelPass {
public:
    OPENVINO_RTTI("RemoveRtInfo");
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};
}  // namespace pass
}  // namespace ov
