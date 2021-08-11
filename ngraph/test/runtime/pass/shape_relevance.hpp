// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/pass/pass.hpp"

namespace ov
{
    namespace pass
    {
        class BACKEND_API ShapeRelevance : public FunctionPass
        {
        public:
            ShapeRelevance()
                : FunctionPass()
            {
            }
            virtual bool run_on_function(std::shared_ptr<ov::Function> f) override;
        };
    } // namespace pass
} // namespace ov
