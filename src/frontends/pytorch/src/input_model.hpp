// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class TranslateSession;

struct PlaceDesc {
    PlaceDesc(const element::Type& type, const PartialShape& pshape)
        : m_type(type),
          m_pshape(pshape),
          m_value(nullptr) {}
    element::Type m_type;
    PartialShape m_pshape;
    std::shared_ptr<Node> m_value;
};

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::pytorch::TranslateSession;
    friend class ::ov::frontend::pytorch::Place;

public:
    // TODO: pass telemetry extension to this ctor
    explicit InputModel(std::shared_ptr<TorchDecoder> model_decoder);

    std::vector<ov::frontend::Place::Ptr> get_inputs() const override ;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
    void set_partial_shape(const Place::Ptr& place, const ov::PartialShape& shape) override;
    ov::PartialShape get_partial_shape(const Place::Ptr& place) const override;
    void set_element_type(const Place::Ptr& place, const ov::element::Type& type) override;
    ov::element::Type get_element_type(const Place::Ptr& place) const override;
    void set_tensor_value(const Place::Ptr& place, const void* value) override;

private:
    std::shared_ptr<TorchDecoder> m_model_decoder;
    std::unordered_map<std::string, std::shared_ptr<frontend::Place>> m_name_to_place;
    std::unordered_map<size_t, PlaceDesc> m_descriptors;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
