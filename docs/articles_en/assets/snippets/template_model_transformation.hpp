// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

class MyModelTransformation;

}  // namespace pass
}  // namespace ov

// ! [model_pass:template_transformation_hpp]
// template_model_transformation.hpp
class ov::pass::MyModelTransformation : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MyModelTransformation");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};
// ! [model_pass:template_transformation_hpp]
