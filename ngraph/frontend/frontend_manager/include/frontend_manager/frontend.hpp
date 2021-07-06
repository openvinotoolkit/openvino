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

            /// \brief Loads an input model by specified model file path
            /// If model is stored in several files (e.g. model topology and model weights) -
            /// frontend implementation is responsible to handle this case, generally frontend may
            /// retrieve other file names from main file
            /// \param path Main model file path
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_file(const std::string& path) const;

            /// \brief Loads an input model by specified number of model files
            /// This shall be used for cases when client knows all model files (model, weights, etc)
            /// \param paths Array of model files
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_files(const std::vector<std::string>& paths) const;

            /// \brief Loads an input model by already loaded memory buffer
            /// Memory structure is frontend-defined and is not specified in generic API
            /// \param model Model memory buffer
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_memory(const void* model) const;

            /// \brief Loads an input model from set of memory buffers
            /// Memory structure is frontend-defined and is not specified in generic API
            /// \param modelParts Array of model memory buffers
            /// \return Loaded input model
            virtual InputModel::Ptr
                load_from_memory_fragments(const std::vector<const void*>& modelParts) const;

            /// \brief Loads an input model by input stream representing main model file
            /// \param stream Input stream of main model
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_stream(std::istream& stream) const;

            /// \brief Loads an input model by input streams representing all model files
            /// \param streams Array of input streams for model
            /// \return Loaded input model
            virtual InputModel::Ptr
                load_from_streams(const std::vector<std::istream*>& streams) const;

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
        };

    } // namespace frontend

} // namespace ngraph
