// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/input_model.hpp>
#include <frontend_manager/place.hpp>
#include <tensorflow_frontend/graph_iterator.hpp>

namespace ov {
namespace frontend {

class OpPlaceTF;
class TensorPlaceTF;

class InputModelTF : public ngraph::frontend::InputModel {
    friend class FrontEndTF;
    class InputModelTFImpl;
    std::shared_ptr<InputModelTFImpl> _impl;

    std::vector<std::shared_ptr<OpPlaceTF>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlaceTF>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

public:
    explicit InputModelTF(const GraphIterator::Ptr& graph_iterator);

    std::vector<ngraph::frontend::Place::Ptr> get_inputs() const override;
    std::vector<ngraph::frontend::Place::Ptr> get_outputs() const override;
    ngraph::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    void override_all_outputs(const std::vector<ngraph::frontend::Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<ngraph::frontend::Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<ngraph::frontend::Place::Ptr>& inputs,
                          const std::vector<ngraph::frontend::Place::Ptr>& outputs) override;
    void set_partial_shape(ngraph::frontend::Place::Ptr place, const ov::PartialShape&) override;
    ov::PartialShape get_partial_shape(ngraph::frontend::Place::Ptr place) const override;
    void set_element_type(ngraph::frontend::Place::Ptr place, const ov::element::Type&) override;
    void set_tensor_value(ngraph::frontend::Place::Ptr place, const void* value) override;
};
}  // namespace frontend
}  // namespace ov
