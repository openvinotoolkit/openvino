// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations/utils/pass_param.hpp>

#include <ngraph/pass/manager.hpp>


namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConversionPassManager);

}  // namespace pass
}  // namespace ngraph


class ngraph::pass::ConversionPassManager : public ::ngraph::pass::Manager, public ::ngraph::pass::PassParam {
public:
    explicit ConversionPassManager(const PassParam::param_callback & callback = PassParam::getDefaultCallback())
            : Manager(), PassParam(callback) {
        register_conversion_passes();
    }

private:
    void register_conversion_passes();
};
