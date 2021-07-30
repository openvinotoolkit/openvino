// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SubtractMeanInputs;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::SubtractMeanInputs : public ngraph::pass::FunctionPass {
public:
    using MeanMap = std::map<std::string, std::shared_ptr<ngraph::op::Constant>>;

    enum class Version {
        IR_V10
    };
    NGRAPH_RTTI_DECLARATION;

    SubtractMeanInputs(const MeanMap& mean_map);

    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

private:
    MeanMap m_mean_map;
};