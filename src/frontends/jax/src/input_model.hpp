// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/place.hpp"

namespace ov {
namespace frontend {
namespace jax {

class TranslateSession;
class Place;
class JaxDecoder;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::jax::TranslateSession;

public:
    explicit InputModel(const std::shared_ptr<JaxDecoder>& model_decoder);

    std::vector<frontend::Place::Ptr> get_inputs() const override;
    std::vector<frontend::Place::Ptr> get_outputs() const override;
    frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override;
    void set_partial_shape(const frontend::Place::Ptr& place, const ov::PartialShape& shape) override;
    ov::PartialShape get_partial_shape(const frontend::Place::Ptr& place) const override;
    void set_element_type(const frontend::Place::Ptr& place, const ov::element::Type& type) override;
    ov::element::Type get_element_type(const frontend::Place::Ptr& place) const override;
    void set_tensor_value(const frontend::Place::Ptr& place, const void* value) override;
    void override_all_outputs(const std::vector<frontend::Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<frontend::Place::Ptr>& inputs) override;
    std::shared_ptr<JaxDecoder> get_decoder() const;

private:
    std::shared_ptr<JaxDecoder> m_model_decoder;
    std::unordered_map<std::string, std::shared_ptr<frontend::Place>> m_name_to_place;
    std::vector<std::shared_ptr<frontend::Place>> m_inputs;
    std::vector<std::shared_ptr<frontend::Place>> m_outputs;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
