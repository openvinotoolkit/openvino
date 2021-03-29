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

// TODO: include it by just frontend_manager.hpp without path
//#include "../../include/frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager.hpp"

namespace tensorflow { class GraphDef; }

namespace ngraph
{
    namespace frontend
    {
        class PlaceTensorflow : public Place
        {
        public:

            std::string name;
            enum Kind { PORT_INPUT, PORT_OUTPUT, TENSOR, OP } kind;
            size_t port;

            PlaceTensorflow (const std::string& _name, Kind _kind = OP, size_t _port = 0) : name(_name), kind(_kind), port(_port) {}
        };

        class NGRAPH_API InputModelTensorflow : public InputModel
        {
        public:

            std::shared_ptr<tensorflow::GraphDef> graph_def;
            std::string path;

            // TODO: map from PlaceTensorflow, not from name string
            std::map<std::string, ngraph::PartialShape> partialShapes;

            InputModelTensorflow (const std::string& _path);

            std::vector<Place::Ptr> getInputs () const override;

            void setPartialShape (Place::Ptr place, const ngraph::PartialShape& pshape) override;
        };

        class NGRAPH_API FrontEndTensorflow : public FrontEnd
        {
        public:

            FrontEndTensorflow ()
            {
            }

            virtual InputModel::Ptr loadFromFile (const std::string& path) const override
            {
                return std::make_shared<InputModelTensorflow>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const override;
        };

    } // namespace frontend

} // namespace ngraph
