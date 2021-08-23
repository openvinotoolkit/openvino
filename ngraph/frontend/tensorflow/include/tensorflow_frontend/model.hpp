// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: include it by just frontend_manager.hpp without path
#include <frontend_manager/frontend.hpp>
#include <tensorflow_frontend/utility.hpp>

namespace tensorflow {
class GraphDef;
class NodeDef;
namespace ngraph_bridge {
class GraphIteratorProto;
}
}  // namespace tensorflow

namespace ngraph {
namespace frontend {

class TF_API InputModelTensorflow : public InputModel {
public:
    // TODO: move these members to private section
    std::shared_ptr<::tensorflow::ngraph_bridge::GraphIteratorProto> graph_impl;
    std::shared_ptr<::tensorflow::GraphDef> graph_def;
    std::string path;
    std::vector<ngraph::PartialShape> input_shapes;
    // TODO: map from PlaceTensorflow, not from name string
    std::map<std::string, ngraph::PartialShape> partialShapes;

public:
    InputModelTensorflow(const std::string& _path);
    InputModelTensorflow(const std::vector<std::istream*>& streams);
    // TODO: remove these constructors
    InputModelTensorflow(std::shared_ptr<::tensorflow::GraphDef> _graph_def,
                         std::vector<ngraph::PartialShape> _input_shapes = {});
    InputModelTensorflow(const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _nodes_def,
                         std::vector<ngraph::PartialShape> _input_shapes = {});

    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override {
        // TODO: implement
        return {};
    }
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override {
        // TODO: implement
        return nullptr;
    }
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override {
        // TODO: implement
    }
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override {
        // TODO: implement
    }
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override {
        // TODO: implement
    }
    virtual void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& pshape) override;
    virtual ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
    void set_element_type(Place::Ptr place, const ngraph::element::Type&) override{
        // TODO: implement
    };
    void set_tensor_value(Place::Ptr place, const void* value) override{
        // TODO: implement
    };
};

}  // namespace frontend

}  // namespace ngraph
