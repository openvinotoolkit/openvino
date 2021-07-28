// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// Root of nodes that can be sink nodes
        class NGRAPH_API Sink : public Op
        {
        public:
            virtual ~Sink() = 0;
            NGRAPH_RTTI_DECLARATION;

        protected:
            Sink()
                : Op()
            {
            }

            explicit Sink(const OutputVector& arguments)
                : Op(arguments)
            {
            }
        };
    } // namespace op
    using SinkVector = std::vector<std::shared_ptr<op::Sink>>;
} // namespace ngraph
