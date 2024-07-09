// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_cpu {

// ! [model_pass:template_transformation_hpp]
// template_model_transformation.hpp
class DuplicateModel : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("DuplicateModel", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};
// ! [model_pass:template_transformation_hpp]

}  // namespace intel_cpu
}  // namespace ov
