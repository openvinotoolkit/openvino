//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include "exceptions.hpp"
#include "model.hpp"

namespace ngraph {
namespace frontend {

class NGRAPH_API FrontEndPDPD : public FrontEnd
{
    static std::shared_ptr<Function> convert_model(const std::shared_ptr<InputModelPDPD>& model);
public:

    FrontEndPDPD () = default;

    /**
     * @brief Reads model from file and deducts file names of weights
     * @param path path to folder which contains __model__ file or path to .pdmodel file
     * @return InputModel::Ptr
     */
    virtual InputModel::Ptr loadFromFile (const std::string& path) const override;

    /**
     * @brief Reads model and weights from files
     * @param paths vector containing path to .pdmodel and .pdiparams files
     * @return InputModel::Ptr
     */
    virtual InputModel::Ptr loadFromFiles (const std::vector<std::string>& paths) const override;

    /**
     * @brief Reads model from stream
     * @param model_stream stream containing .pdmodel or __model__ files. Can only be used if model have no weights
     * @return InputModel::Ptr
     */
    virtual InputModel::Ptr loadFromStream (std::istream& model_stream) const override;

    /**
     * @brief Reads model from stream
     * @param paths vector of streams containing .pdmodel and .pdiparams files. Can't be used in case of multiple weight files
     * @return InputModel::Ptr
     */
    virtual InputModel::Ptr loadFromStreams (const std::vector<std::istream*>& paths) const override;

    std::shared_ptr<Function> convert (InputModel::Ptr model) const override;
};

} // namespace frontend
} // namespace ngraph
