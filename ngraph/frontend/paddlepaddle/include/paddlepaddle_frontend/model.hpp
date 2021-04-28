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

#include <frontend_manager/ifrontend_manager.hpp>
#include "utility.hpp"

namespace ngraph {
namespace frontend {

class OpPlacePDPD;
class TensorPlacePDPD;

class InputModelPDPD : public IInputModel
{
    friend class FrontEndPDPD;
    class InputModelPDPDImpl;
    std::shared_ptr<InputModelPDPDImpl> _impl;

    std::vector<uint8_t> readWeight(const std::string& name, int64_t len);
    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces(int i) const;
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces(int i) const;
    size_t getBlockNumber() const;
    std::map<std::string, Output<Node>> getTensorValues() const;

public:
    explicit InputModelPDPD (const std::string& _path);
    std::vector<IPlace::Ptr> getInputs () const override;
    std::vector<IPlace::Ptr> getOutputs () const override;
    IPlace::Ptr getPlaceByTensorName (const std::string& tensorName) const override;
    void overrideAllOutputs (const std::vector<IPlace::Ptr>& outputs) override;
    void overrideAllInputs (const std::vector<IPlace::Ptr>& inputs) override;
    void extractSubgraph (const std::vector<IPlace::Ptr>& inputs, const std::vector<IPlace::Ptr>& outputs) override;
    void setDefaultShape (IPlace::Ptr place, const ngraph::Shape&) override;
    void setPartialShape (IPlace::Ptr place, const ngraph::PartialShape&) override;
    void setElementType (IPlace::Ptr place, const ngraph::element::Type&) override;
    void setTensorValue (IPlace::Ptr place, const void* value) override;

};

} // namespace frontend
} // namespace ngraph
