// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"

#include <utility>
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class OpPlace;
class TensorPlace;

class InputModel : public ov::frontend::InputModel {
    friend class FrontEnd;
protected:
    class InputModelTFImpl {
    public:
        InputModelTFImpl(const GraphIterator::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
        InputModelTFImpl(const GraphIterator::Ptr& graph_iterator,
                         const ov::frontend::InputModel& input_model,
                         const std::shared_ptr<TelemetryExtension>& telemetry);
        virtual std::vector<ov::frontend::Place::Ptr> getInputs() const;
        virtual std::vector<ov::frontend::Place::Ptr> getOutputs() const;
        virtual ov::frontend::Place::Ptr getPlaceByTensorName(const std::string& tensorName) const;
        virtual void overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
        virtual void overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
        virtual void extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                     const std::vector<ov::frontend::Place::Ptr>& outputs);
        virtual void setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape&);
        virtual ov::PartialShape getPartialShape(ov::frontend::Place::Ptr place) const;
        virtual void setElementType(ov::frontend::Place::Ptr place, const ov::element::Type&);
        virtual ov::element::Type getElementType(ov::frontend::Place::Ptr place) const;
        virtual void setTensorValue(ov::frontend::Place::Ptr place, const void* value);
        virtual std::vector<std::shared_ptr<OpPlace>> get_op_places() const;

        virtual std::map<std::string, std::shared_ptr<TensorPlace>> get_tensor_places() const {
            return m_tensor_places;
        }

        virtual std::map<std::string, Output<Node>> get_tensor_values() const {
            return m_tensor_values;
        };

    private:
        void loadPlaces();
        std::vector<std::shared_ptr<OpPlace>> determine_cut_nodes() const;

        std::vector<std::shared_ptr<OpPlace>> m_op_places;
        std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
        mutable std::map<std::string, std::shared_ptr<TensorPlace>> m_tensor_places;
        std::vector<ov::frontend::Place::Ptr> m_inputs;
        std::vector<ov::frontend::Place::Ptr> m_outputs;
        std::map<std::string, Output<Node>> m_tensor_values;

        std::shared_ptr<GraphIterator> m_graph_iterator;
        const ov::frontend::InputModel& m_input_model;

        std::shared_ptr<TelemetryExtension> m_telemetry;

        // shows if some nodes might be deleted from graph
        bool m_graph_changed = false;
    };
    std::shared_ptr<InputModelTFImpl> _impl;
    explicit InputModel(std::shared_ptr<InputModelTFImpl> impl) : _impl{std::move(impl)} {}
private:
    std::vector<std::shared_ptr<OpPlace>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlace>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

public:
    explicit InputModel(const GraphIterator::Ptr& graph_iterator,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});

    std::vector<ov::frontend::Place::Ptr> get_inputs() const override;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const override;
    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                          const std::vector<ov::frontend::Place::Ptr>& outputs) override;
    void set_partial_shape(const ov::frontend::Place::Ptr& place, const ov::PartialShape&) override;
    ov::PartialShape get_partial_shape(const ov::frontend::Place::Ptr& place) const override;
    void set_element_type(const ov::frontend::Place::Ptr& place, const ov::element::Type&) override;
    ov::element::Type get_element_type(const ov::frontend::Place::Ptr& place) const override;
    void set_tensor_value(const ov::frontend::Place::Ptr& place, const void* value) override;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
