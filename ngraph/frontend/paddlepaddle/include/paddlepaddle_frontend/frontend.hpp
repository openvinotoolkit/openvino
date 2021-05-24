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

            /**
             * @brief Reads model from file and deducts file names of weights
             * @param path path to folder which contains __model__ file or path to .pdmodel file
             * @return InputModel::Ptr
             */
            InputModel::Ptr load_from_file(const std::string& path) const override;

            /**
             * @brief Reads model and weights from files
             * @param paths vector containing path to .pdmodel and .pdiparams files
             * @return InputModel::Ptr
             */
            InputModel::Ptr load_from_files(const std::vector<std::string>& paths) const override;

            /**
             * @brief Reads model from stream
             * @param model_stream stream containing .pdmodel or __model__ files. Can only be used
             * if model have no weights
             * @return InputModel::Ptr
             */
            InputModel::Ptr load_from_stream(std::istream& model_stream) const override;

            /**
             * @brief Reads model from stream
             * @param paths vector of streams containing .pdmodel and .pdiparams files. Can't be
             * used in case of multiple weight files
             * @return InputModel::Ptr
             */
            InputModel::Ptr
                load_from_streams(const std::vector<std::istream*>& paths) const override;

            std::shared_ptr<Function> convert(InputModel::Ptr model) const override;
        };

    } // namespace frontend
} // namespace ngraph
