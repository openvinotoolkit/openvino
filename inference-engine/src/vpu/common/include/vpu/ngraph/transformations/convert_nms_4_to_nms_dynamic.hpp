// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class UpgradeNMS4ToNMSDynamic : public ngraph::pass::GraphRewrite {
public:
    UpgradeNMS4ToNMSDynamic() : GraphRewrite() {
        upgrade_nms4_to_nms_dynamic();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    void upgrade_nms4_to_nms_dynamic();
};

} // namespace vpu
