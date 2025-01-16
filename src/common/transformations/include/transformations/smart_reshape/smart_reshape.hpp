// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SmartReshape;

}  // namespace pass
}  // namespace ov

class ov::pass::SmartReshape : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SmartReshape", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
