// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/paddle/frontend.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class OpPlace;
class TensorPlace;

class InputModel : public ov::frontend::InputModel {
public:
    explicit InputModel(const std::string& path, const std::shared_ptr<TelemetryExtension>& telemetry = {});
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    explicit InputModel(const std::wstring& path, const std::shared_ptr<TelemetryExtension>& telemetry = {});
#endif
    explicit InputModel(const std::vector<std::istream*>& streams,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});
    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void set_partial_shape(const Place::Ptr& place, const ov::PartialShape&) override;
    ov::PartialShape get_partial_shape(const Place::Ptr& place) const override;
    void set_element_type(const Place::Ptr& place, const ov::element::Type&) override;
    void set_tensor_value(const Place::Ptr& place, const void* value) override;

private:
    friend class ov::frontend::paddle::FrontEnd;
    class InputModelImpl;
    std::shared_ptr<InputModelImpl> _impl;

    std::vector<std::shared_ptr<OpPlace>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlace>> get_var_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
