// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "frontend_manager_defs.hpp"
#include "input_model.hpp"
#include "ngraph/function.hpp"
#include "ngraph/variant.hpp"

namespace ngraph
{
    namespace frontend
    {
        /// \brief An interface for identifying a frontend for a particular framework.
        /// Provides an ability to load and convert of input model
        class FRONTEND_API FrontEnd
        {
        public:
            typedef std::shared_ptr<FrontEnd> Ptr;

            FrontEnd();

            virtual ~FrontEnd();

            /// \brief Validates if FrontEnd can recognize model with parameters specified.
            /// Same parameters should be used to load model.
            /// \param vars Any number of parameters of any type. What kind of parameters
            /// are accepted is determined by each FrontEnd individually, typically it is
            /// std::string containing path to the model file. For more information please
            /// refer to specific FrontEnd documentation.
            /// \return true if model recognized, false - otherwise.
            template <typename... Types>
            inline bool supported(const Types&... vars) const
            {
                return supported_impl({make_variant(vars)...});
            }

            /// \brief Loads an input model by any specified arguments. Each FrontEnd separately
            /// defines what arguments it can accept.
            /// \param vars Any number of parameters of any type. What kind of parameters
            /// are accepted is determined by each FrontEnd individually, typically it is
            /// std::string containing path to the model file. For more information please
            /// refer to specific FrontEnd documentation.
            /// \return Loaded input model.
            template <typename... Types>
            inline InputModel::Ptr load(const Types&... vars) const
            {
                return load_impl({make_variant(vars)...});
            }

            /// \brief Completely convert and normalize entire function, throws if it is not
            /// possible
            /// \param model Input model
            /// \return fully converted nGraph function
            virtual std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const;

            /// \brief Completely convert the remaining, not converted part of a function.
            /// \param partiallyConverted partially converted nGraph function
            /// \return fully converted nGraph function
            virtual std::shared_ptr<ngraph::Function>
                convert(std::shared_ptr<ngraph::Function> partiallyConverted) const;

            /// \brief Convert only those parts of the model that can be converted leaving others
            /// as-is. Converted parts are not normalized by additional transformations; normalize
            /// function or another form of convert function should be called to finalize the
            /// conversion process.
            /// \param model Input model
            /// \return partially converted nGraph function
            virtual std::shared_ptr<ngraph::Function>
                convert_partially(InputModel::Ptr model) const;

            /// \brief Convert operations with one-to-one mapping with decoding nodes.
            /// Each decoding node is an nGraph node representing a single FW operation node with
            /// all attributes represented in FW-independent way.
            /// \param model Input model
            /// \return nGraph function after decoding
            virtual std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const;

            /// \brief Runs normalization passes on function that was loaded with partial conversion
            /// \param function partially converted nGraph function
            virtual void normalize(std::shared_ptr<ngraph::Function> function) const;

        protected:
            virtual bool
                supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const;
            virtual InputModel::Ptr
                load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const;
        };

        template <>
        inline bool FrontEnd::supported(const std::vector<std::shared_ptr<Variant>>& variants) const
        {
            return supported_impl(variants);
        }

    } // namespace frontend

} // namespace ngraph
