// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>

namespace ngraph
{
    namespace frontend
    {
        class FRONTEND_API FrontEndONNX : public FrontEnd
        {
        public:
            std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override;
            std::shared_ptr<ngraph::Function>
                convert(std::shared_ptr<ngraph::Function> partially_converted) const override;
            std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const override;
        protected:
            InputModel::Ptr
                load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override;


        };

    } // namespace frontend

} // namespace ngraph
