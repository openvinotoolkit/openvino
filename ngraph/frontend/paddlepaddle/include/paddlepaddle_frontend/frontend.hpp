// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include "exceptions.hpp"
#include "model.hpp"

namespace ngraph
{
    namespace frontend
    {
        class PDPD_API FrontEndPDPD : public FrontEnd
        {
            static std::shared_ptr<Function>
                convert_model(const std::shared_ptr<InputModelPDPD>& model);

        public:
            FrontEndPDPD() = default;

            bool supported_by_variants(
                const std::vector<std::shared_ptr<Variant>>& variants) const override;

            std::shared_ptr<Function> convert(InputModel::Ptr model) const override;

        protected:
            /**
             * @brief Reads model from 1 or 2 given file names or 1 or 2 std::istream containing
             * model in protobuf format and weights
             * @param params Can be path to folder which contains __model__ file or path to .pdmodel
             * file
             * @return InputModel::Ptr
             */
            InputModel::Ptr
                load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override;
        };

    } // namespace frontend
} // namespace ngraph
