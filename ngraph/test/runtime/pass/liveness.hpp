// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class Liveness;
    }
}

class BACKEND_API ngraph::pass::Liveness : public FunctionPass
{
public:
    bool run_on_function(std::shared_ptr<ngraph::Function>) override;
};
