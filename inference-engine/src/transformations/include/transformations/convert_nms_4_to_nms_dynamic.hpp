// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API UpgradeNMS4ToNMSDynamic;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      UpgradeNMS4ToNMSDynamic transformation upgrades NonMaxSuppression operations from v3 version to dynamic
 *      in case function has at least one NonZero operation (dynamism marker).
 *      NMS dynamic always has dynamic output
 */


class ngraph::pass::UpgradeNMS4ToNMSDynamic: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

