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

class TRANSFORMATIONS_API UpgradeNMS3ToNMS4;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      UpgradeNMS3ToNMS4 transformation upgrades NonMaxSuppression operations from v3 version to v4
 *      in case function has at least one NonZero operation (dynamism marker).
 *      NMS of version v4 always has dynamic output
 */


class ngraph::pass::UpgradeNMS3ToNMS4: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

