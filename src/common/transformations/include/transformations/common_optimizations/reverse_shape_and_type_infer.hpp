// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReverseShapeAndTypeInfer;

}  // namespace pass
}  // namespace ov

/**
 * @brief Perform reverse shape and type infer to duduce input rank and type in certain cases
 */
class ov::pass::ReverseShapeAndTypeInfer : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ReverseShapeAndTypeInfer", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};
