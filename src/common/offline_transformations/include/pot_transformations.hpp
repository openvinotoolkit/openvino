// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>

namespace ngraph {
namespace pass {

class POTTransformations;

}  // namespace pass
}  // namespace ngraph

/**
 * @brief This transformation is an entry point for nGraph transformations that will be
 * executed inside POT.
 */

class ngraph::pass::POTTransformations : public ngraph::pass::FunctionPass {
    std::string m_device;

public:
    OPENVINO_RTTI("POTTransformations", "0");
    explicit POTTransformations(std::string device) : m_device(std::move(device)) {}

    bool run_on_model(const std::shared_ptr<ngraph::Function>&) override;
};
