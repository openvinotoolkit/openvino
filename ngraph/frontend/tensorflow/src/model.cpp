// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/model.hpp>
#include <tensorflow_frontend/place.hpp>

#include <numeric>
#include <fstream>

//#include "graph.pb.h"
//#include "tensor.pb.h"

#include <ngraph/pass/manager.hpp>

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"
//#include <ngraph/pass/transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>

#include "default_opset.h"
#include "graph.hpp"
#include "ngraph_builder.h"
#include "ngraph_conversions.h"

using namespace google;

using namespace ngraph::frontend;

using ::tensorflow::GraphDef;
using ::tensorflow::ngraph_bridge::GraphIteratorProto;

InputModelTensorflow::InputModelTensorflow(const std::string& _path) : path(_path) {
    std::ifstream pb_stream(path, std::ios::binary);
    graph_def = std::make_shared<GraphDef>();
    std::cout << "[ INFO ] Model Parsed: " << graph_def->ParseFromIstream(&pb_stream) << std::endl;
    std::cout << "[ INFO ] Loaded model contains " << graph_def->node_size() << " nodes." << std::endl;
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(graph_def.get());
}

InputModelTensorflow::InputModelTensorflow(std::shared_ptr<::tensorflow::GraphDef> _graph_def,
                                           std::vector<ngraph::PartialShape> _input_shapes)
    : input_shapes(_input_shapes) {
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(_graph_def.get());
}

InputModelTensorflow::InputModelTensorflow(const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _nodes_def,
                                           std::vector<ngraph::PartialShape> _input_shapes)
    : input_shapes(_input_shapes) {
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(_nodes_def);
}

std::vector<Place::Ptr> InputModelTensorflow::get_inputs() const {
    std::vector<Place::Ptr> result;
    for (; !graph_impl->is_end(); graph_impl->next()) {
        std::cout << "graph_impl->get()->op() = " << graph_impl->get()->op() << "\n";
        if (graph_impl->get()->op() == "Placeholder")
            result.push_back(std::make_shared<PlaceTensorflow>(graph_impl->get()->name()));
    }
    graph_impl->reset();
    return result;
}

void InputModelTensorflow::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& pshape) {
    auto place_tf = std::dynamic_pointer_cast<PlaceTensorflow>(place);
    partialShapes[place_tf->name] = pshape;
}

ngraph::PartialShape InputModelTensorflow::get_partial_shape(Place::Ptr place) const {
    auto place_tf = std::dynamic_pointer_cast<PlaceTensorflow>(place);
    ngraph::PartialShape result_shape;
    // TODO: replace by node cache without going through all nodes each time
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto node = graph_impl->get();
        if (node->name() == place_tf->name) {
            node->getAttrValue2("shape", &result_shape);
            break;
        }
    }
    // WARNING! Redesign GraphIterator -- it is not really good thing, detach an iterator from graph itself
    graph_impl->reset();
    return result_shape;
}
