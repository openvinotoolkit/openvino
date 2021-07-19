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
        public:
            FrontEndPDPD() = default;

            /// \brief Completely convert the remaining, not converted part of a function.
            /// \param partiallyConverted partially converted nGraph function
            /// \return fully converted nGraph function
            std::shared_ptr<Function> convert(InputModel::Ptr model) const override;

        protected:
            /// \brief Check if FrontEndPDPD can recognize model from given parts
            /// \param params Can be path to folder which contains __model__ file or path to
            /// .pdmodel file
            /// \return InputModel::Ptr
            bool supported_impl(
                const std::vector<std::shared_ptr<Variant>>& variants) const override;

            /// \brief Reads model from 1 or 2 given file names or 1 or 2 std::istream containing
            /// model in protobuf format and weights
            /// \param params Can contain path to folder with __model__ file or path to .pdmodel
            /// file or 1 or 2 streams with model and weights
            /// \return InputModel::Ptr
            InputModel::Ptr
                load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override;

        private:
            static std::shared_ptr<Function>
                convert_model(const std::shared_ptr<InputModelPDPD>& model);
        };

    } // namespace frontend
} // namespace ngraph
