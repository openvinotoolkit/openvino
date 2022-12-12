// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "input_model.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
class OpPlace;
class TensorPlace;
}
namespace tensorflow_lite {
class InputModel : public ov::frontend::tensorflow::InputModel {
    friend class FrontEnd;
    class InputModelTFLiteImpl : public ov::frontend::tensorflow::InputModel::InputModelTFImpl {
    public:
        InputModelTFLiteImpl(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
        InputModelTFLiteImpl(const tensorflow::GraphIterator::Ptr& graph_iterator,
                             const ov::frontend::InputModel& input_model,
                             const std::shared_ptr<TelemetryExtension>& telemetry);

        ov::frontend::Place::Ptr getPlaceByTensorName(const std::string& tensorName) const override;
        void overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs) override;
        void overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs) override;
        void extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                             const std::vector<ov::frontend::Place::Ptr>& outputs) override;
        void setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape&) override;
        ov::PartialShape getPartialShape(ov::frontend::Place::Ptr place) const override;
        void setElementType(ov::frontend::Place::Ptr place, const ov::element::Type&) override;
        ov::element::Type getElementType(ov::frontend::Place::Ptr place) const override;
        void setTensorValue(ov::frontend::Place::Ptr place, const void* value) override;

        std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> get_op_places() const override;
        std::map<std::string, std::shared_ptr<ov::frontend::tensorflow::TensorPlace>> get_tensor_places() const override {
            FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::get_tensor_places");
            return {};
        }
        std::map<std::string, Output<Node>> get_tensor_values() const override {
            FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::get_tensor_values");
            return {};
        };

    private:
        void loadPlaces();
        std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> determine_cut_nodes() const;
    };
    std::shared_ptr<InputModelTFLiteImpl> _impl;
    std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> get_op_places() const;
    std::map<std::string, std::shared_ptr<ov::frontend::tensorflow::TensorPlace>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

public:
    explicit InputModel(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
