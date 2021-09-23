// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ngraph/opsets/opset8.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation replaces the provided pairs Parameter and Result with ngraph Memory layers
 * ReadValue and Assign
 */
class OPENVINO_API MakeStateful : public FunctionPass {
public:
    OPENVINO_RTTI("MakeStateful");

    using ParamResPairs =
        std::vector<std::pair<std::shared_ptr<ngraph::opset8::Parameter>, std::shared_ptr<ngraph::opset8::Result>>>;

    static ParamResPairs find_param_results_by_names(const std::shared_ptr<ngraph::Function>& func,
                                                     const std::map<std::string, std::string>& param_res_names);
    explicit MakeStateful(const ParamResPairs& pairs_to_replace) : m_pairs_to_replace(pairs_to_replace) {}
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    ParamResPairs m_pairs_to_replace;
};
}  // namespace pass
}  // namespace ov
