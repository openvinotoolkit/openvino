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

namespace ngraph {
namespace frontend {

class PlacePDPD : public Place
{
    // TODO
};

class InPortPlacePDPD : public PlacePDPD
{
    // TODO
};

class OutPortPlacePDPD : public PlacePDPD
{
    // TODO
};

class VarPlacePDPD;
class OpPlacePDPD : public PlacePDPD
{
public:
    InputModel* model;
    const void* op; // TODO: make it cleaner
    std::map<std::string, std::vector<std::weak_ptr<VarPlacePDPD>>> outputs;
    std::map<std::string, std::vector<std::weak_ptr<VarPlacePDPD>>> inputs;
    OpPlacePDPD(const void* _op) : op(_op) {}
};

class VarPlacePDPD : public PlacePDPD
{
public:
    InputModel* model;
    const void* var; // TODO: make it cleaner
    std::vector<std::weak_ptr<OpPlacePDPD>> producing_ops; // should never have more than 1 element
    std::vector<std::weak_ptr<OpPlacePDPD>> consuming_ops;
    PartialShape shape;
    element::Type type;
    VarPlacePDPD(const void* _var) : var(_var) {}
};

} // namespace frontend
} // namespace ngraph
