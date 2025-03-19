// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/place.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class FrontEnd;
class TranslateSession;
class Place;
class TorchDecoder;

struct PlaceDesc {
    PlaceDesc(const std::shared_ptr<Node>& value) : m_value(value) {}
    std::shared_ptr<Node> m_value;
};

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::pytorch::TranslateSession;
    friend class ::ov::frontend::pytorch::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<TorchDecoder>& model_decoder);

    std::vector<frontend::Place::Ptr> get_inputs() const override;
    std::vector<frontend::Place::Ptr> get_outputs() const override;
    frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
    frontend::Place::Ptr get_place_by_input_index(size_t input_idx) const override;
    void set_partial_shape(const frontend::Place::Ptr& place, const ov::PartialShape& shape) override;
    ov::PartialShape get_partial_shape(const frontend::Place::Ptr& place) const override;
    void set_element_type(const frontend::Place::Ptr& place, const ov::element::Type& type) override;
    ov::element::Type get_element_type(const frontend::Place::Ptr& place) const override;
    void set_tensor_value(const frontend::Place::Ptr& place, const void* value) override;
    void override_all_outputs(const std::vector<frontend::Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<frontend::Place::Ptr>& inputs) override;
    const std::string& decoder_type_name() const;
    std::shared_ptr<TorchDecoder> get_decoder() const;
    // update input places and erase requested places if possible
    void flush_places();

private:
    std::shared_ptr<TorchDecoder> m_model_decoder;
    std::unordered_map<std::string, std::shared_ptr<frontend::Place>> m_name_to_place;
    std::vector<std::shared_ptr<frontend::Place>> m_inputs;
    std::vector<std::shared_ptr<frontend::Place>> m_outputs;
    std::vector<std::shared_ptr<frontend::Place>> m_requested_places;
    std::unordered_map<size_t, PlaceDesc> m_descriptors;
    const std::string m_decoder_type_name;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
