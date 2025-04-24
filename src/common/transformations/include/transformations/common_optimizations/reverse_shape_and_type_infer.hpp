// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReverseShapeAndTypeInfer;

}  // namespace pass
}  // namespace ov

/**
 * @brief Perform reverse shape and type infer to deduce input rank and type in certain cases
 */
class ov::pass::ReverseShapeAndTypeInfer : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ReverseShapeAndTypeInfer");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

private:
    bool inherit_output_shape(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& input_idxs);
    bool inherit_output_rank(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& input_idxs);
    bool inherit_output_type(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& input_idxs);
};
