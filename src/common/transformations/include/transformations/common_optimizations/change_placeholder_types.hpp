// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ChangePlaceholderTypes;

/**
 * @brief Add OldApiMap with legacy type for Parameter node
 */
class ChangePlaceholderTypes : public ModelPass {
public:
    OPENVINO_RTTI("ChangePlaceholderTypes", "0");
    explicit ChangePlaceholderTypes(const std::vector<std::string>& params_with_custom_types)
        : m_params_with_custom_types(params_with_custom_types) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::vector<std::string> m_params_with_custom_types;
};

}  // namespace pass
}  // namespace ov
