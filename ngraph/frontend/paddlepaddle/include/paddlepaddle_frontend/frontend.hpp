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
        class OpPlacePDPD;

        class PDPD_API FrontEndPDPD : public FrontEnd
        {
        public:
            FrontEndPDPD() = default;

            /// \brief Completely convert the remaining, not converted part of a function.
            /// \param partiallyConverted partially converted nGraph function
            /// \return fully converted nGraph function
            std::shared_ptr<Function> convert(InputModel::Ptr model) const override;

            /// \brief Completely convert the remaining, not converted part of a function.
            /// \param partiallyConverted partially converted nGraph function
            void convert(std::shared_ptr<Function> partiallyConverted) const override;

            /// \brief Convert only those parts of the model that can be converted leaving others
            /// as-is. Converted parts are not normalized by additional transformations; normalize
            /// function or another form of convert function should be called to finalize the
            /// conversion process.
            /// \param model Input model
            /// \return partially converted nGraph function
            std::shared_ptr<Function> convert_partially(InputModel::Ptr model) const override;

            /// \brief Convert operations with one-to-one mapping with decoding nodes.
            /// Each decoding node is an nGraph node representing a single FW operation node with
            /// all attributes represented in FW-independent way.
            /// \param model Input model
            /// \return nGraph function after decoding
            std::shared_ptr<Function> decode(InputModel::Ptr model) const override;

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
                convert_each_node(const std::shared_ptr<InputModelPDPD>& model,
                                  std::function<std::map<std::string, OutputVector>(
                                      const std::map<std::string, Output<Node>>&,
                                      const std::shared_ptr<OpPlacePDPD>&)> func);
        };

    } // namespace frontend
} // namespace ngraph
