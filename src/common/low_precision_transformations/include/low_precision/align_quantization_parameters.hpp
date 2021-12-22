// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/pass/pass.hpp>
#include "openvino/core/ov_visibility.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class OPENVINO_API AlignQuantizationParameters;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::AlignQuantizationParameters : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
