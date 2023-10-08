// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

class POTTransformations;

}  // namespace pass
}  // namespace ov

/**
 * @brief This transformation is an entry point for OpenVINO transformations that will be
 * executed inside POT.
 */

class ov::pass::POTTransformations : public ov::pass::ModelPass {
    std::string m_device;

public:
    OPENVINO_RTTI("POTTransformations", "0");
    explicit POTTransformations(std::string device) : m_device(std::move(device)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};
