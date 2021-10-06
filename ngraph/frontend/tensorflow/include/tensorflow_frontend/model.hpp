// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/input_model.hpp>
#include <frontend_manager/place.hpp>
#include <tensorflow_frontend/graph_iterator.hpp>
#include <tensorflow_frontend/utility.hpp>

namespace ngraph {
namespace frontend {

class OpPlaceTF;
class TensorPlaceTF;

class TF_API InputModelTF : public InputModel {
    friend class FrontEndTF;
    class InputModelTFImpl;
    std::shared_ptr<InputModelTFImpl> _impl;

    std::vector<std::shared_ptr<OpPlaceTF>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlaceTF>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

public:
    explicit InputModelTF(const GraphIterator::Ptr& graph_iterator);

    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void set_partial_shape(Place::Ptr place, const ngraph::PartialShape&) override;
    ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
    void set_element_type(Place::Ptr place, const ngraph::element::Type&) override;
    void set_tensor_value(Place::Ptr place, const void* value) override;
};
}  // namespace frontend
}  // namespace ngraph
