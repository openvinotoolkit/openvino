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

#include "model.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph {
namespace frontend {

class NGRAPH_API FrontEndPDPD : public FrontEnd
{
    std::shared_ptr<Function> convert_model(std::shared_ptr<InputModelPDPD> model) const;
    std::shared_ptr<opset6::Constant> read_tensor(std::shared_ptr<VarPlacePDPD> place,
                                                  std::shared_ptr<InputModelPDPD> model) const;
public:

    FrontEndPDPD ()
    {
    }

    virtual InputModel::Ptr loadFromFile (const std::string& path) const override
    {
        return std::make_shared<InputModelPDPD>(path);
    }

    virtual std::shared_ptr<Function> convert (InputModel::Ptr model) const override;
};

} // namespace frontend
} // namespace ngraph
