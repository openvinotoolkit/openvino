// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"

namespace ov::npuw {

class AddPositionIdsParam : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::AddPositionIdsParam");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace ov::npuw
