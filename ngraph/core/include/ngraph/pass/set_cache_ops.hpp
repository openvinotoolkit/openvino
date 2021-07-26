// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph
{
    namespace pass
    {
        class NGRAPH_API SetCacheOps : public ngraph::pass::FunctionPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            explicit SetCacheOps(bool cache_ops)
                    : m_cache_ops(cache_ops)
            {
            }

            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
        private:
            bool m_cache_ops;
        };
    } // namespace pass
} // namespace ngraph
