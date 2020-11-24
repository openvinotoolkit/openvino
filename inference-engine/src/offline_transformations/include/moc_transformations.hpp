// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <cpp/ie_cnn_network.h>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <ngraph/pass/constant_folding.hpp>

//namespace ngraph {
//namespace pass {

class TRANSFORMATIONS_API MOCTransformations;

//}  // namespace pass
//}  // namespace ngraph

class MOCTransformations: public ngraph::pass::FunctionPass {
    bool m_cf;
public:
    NGRAPH_RTTI_DECLARATION;
    explicit MOCTransformations(bool cf) : m_cf(cf) {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        std::cout << "Hello MOC!" << std::endl;
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        if (!m_cf) {
            auto pass_config = manager.get_pass_config();
            pass_config->disable<ngraph::pass::ConstantFolding>();
        }
        manager.run_passes(f);
        return true;
    }
};
