// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph
{
    namespace pass
    {
        class NGRAPH_API ConvertFP32ToFP16 : public ngraph::pass::GraphRewrite
        {
        public:
            NGRAPH_RTTI_DECLARATION;
            ConvertFP32ToFP16()
                : GraphRewrite()
            {
                convert_constants_precision();
                convert_parameters_precision();
            }

        private:
            void convert_constants_precision();

            void convert_parameters_precision();
        };
    } // namespace pass
} // namespace ngraph
