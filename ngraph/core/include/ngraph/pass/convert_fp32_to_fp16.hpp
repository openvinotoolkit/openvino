// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph
{
    namespace pass
    {
        class NGRAPH_API ConvertFP32ToFP16 : public ngraph::pass::FunctionPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;
            bool run_on_function(std::shared_ptr<ngraph::Function>) override;
        };
    } // namespace pass
} // namespace ngraph
