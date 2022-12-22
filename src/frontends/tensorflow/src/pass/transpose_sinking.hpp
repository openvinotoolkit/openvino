// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

class TransposeSinking : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TransposeSinking");
    TransposeSinking() {
        set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_model(const std::shared_ptr<ov::Model>& function) override;
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
