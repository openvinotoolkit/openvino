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


#include <fstream>
#include "graph.pb.h"

#include "../include/tensorflow_frontend/tensorflow.hpp"

#include "ngraph_builder.h"

using namespace google;

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndTensorflow::convert (InputModel::Ptr model) const
{
    std::string path = std::dynamic_pointer_cast<ngraph::frontend::InputModelTensorflow>(model)->path;
    std::cerr << "[ INFO ] FrontEndTensorflow::convert invoked\n";
    tensorflow::GraphDef fw_model;
    std::ifstream pb_stream(path, std::ios::binary);
    std::cout << "[ INFO ] Model Parsed: " << fw_model.ParseFromIstream(&pb_stream) << std::endl;
    std::cout << "[ INFO ] Loaded model contains " << fw_model.node_size() << " nodes." << std::endl;
    std::shared_ptr<ngraph::Function> f;
    tensorflow::ngraph_bridge::Builder::TranslateGraph({}, {}, &fw_model, "here_should_be_a_graph_name", f);
    //auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{}, ngraph::ParameterVector{});
    std::cerr << "[ ERROR ] Convetion functionality is not implemented; an empty function will be returned.";
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
    return f;
}
