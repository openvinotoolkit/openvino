// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/opsets/opset1.hpp"
#include "tensor_lite_place.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class InputModel : public ov::frontend::InputModel {
    friend class ov::frontend::tensorflow_lite::FrontEnd;
    class InputModelTFLiteImpl;
    std::shared_ptr<InputModelTFLiteImpl> _impl;

    std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> get_op_places() const;
    std::map<std::string, std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

    ////// Subgraph Handling /////
    std::vector<std::shared_ptr<InputModel>> get_subgraphs() const;

public:
    explicit InputModel(const ov::frontend::tensorflow_lite::GraphIterator::Ptr& graph_iterator,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});

    /////  Searching for places  /////
    std::vector<ov::frontend::Place::Ptr> get_inputs() const override;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const override;
    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    ov::frontend::Place::Ptr get_place_by_input_index(size_t input_idx) const override;

    ///// Naming and annotation  /////
    void set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) override;
    void add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) override;
    void set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) override;

    ///// Setting / getting tensor properties  /////
    void set_partial_shape(const Place::Ptr& place, const ov::PartialShape& shape) override;
    ov::PartialShape get_partial_shape(const Place::Ptr& place) const override;
    void set_element_type(const Place::Ptr& place, const ov::element::Type& type) override;
    ov::element::Type get_element_type(const Place::Ptr& place) const override;
    void set_tensor_value(const Place::Ptr& place, const void* value) override;

    ///// Topology Editing  /////
    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                          const std::vector<ov::frontend::Place::Ptr>& outputs) override;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
