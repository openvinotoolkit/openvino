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
            InputModel::Ptr load_from_file(const std::string& path) const override;
            std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override;
            std::shared_ptr<ngraph::Function>
                convert(std::shared_ptr<ngraph::Function> partially_converted) const override;
            std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const override;
        };

    } // namespace frontend

} // namespace ngraph
