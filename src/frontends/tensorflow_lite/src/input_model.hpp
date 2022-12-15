// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/opsets/opset1.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "input_model.hpp"
#include "place.hpp"
#include "graph_iterator_flatbuffer.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class InputModel : public ov::frontend::InputModel {
    friend class ov::frontend::tensorflow_lite::FrontEnd;
    class InputModelTFLiteImpl;
    std::shared_ptr<InputModelTFLiteImpl> _impl;

    std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> get_op_places() const;
    std::map<std::string, std::shared_ptr<ov::frontend::tensorflow::TensorPlace>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

public:
    explicit InputModel(const ov::frontend::tensorflow_lite::GraphIteratorFlatBuffer::Ptr& graph_iterator,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});

    std::vector<ov::frontend::Place::Ptr> get_inputs() const override;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const override;
//    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
//    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) override;
//    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) override;
//    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
//                          const std::vector<ov::frontend::Place::Ptr>& outputs) override;
//    void set_partial_shape(const ov::frontend::Place::Ptr& place, const ov::PartialShape&) override;
//    ov::PartialShape get_partial_shape(const ov::frontend::Place::Ptr& place) const override;
//    void set_element_type(const ov::frontend::Place::Ptr& place, const ov::element::Type&) override;
//    ov::element::Type get_element_type(const ov::frontend::Place::Ptr& place) const override;
//    void set_tensor_value(const ov::frontend::Place::Ptr& place, const void* value) override;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
