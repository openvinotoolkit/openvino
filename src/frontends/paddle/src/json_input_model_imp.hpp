// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "tensor_place.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class JSONInputModel : public ov::frontend::InputModel {
public:
    using Ptr = std::shared_ptr<JSONInputModel>;

    JSONInputModel(const std::string& json_path,
                   const std::string& yml_path,
                   const std::string& params_path,
                   const std::shared_ptr<TelemetryExtension>& telemetry = nullptr);

    // Implement required InputModel interface
    std::vector<Place::Ptr> get_inputs() const override {
        return m_inputs;
    }
    std::vector<Place::Ptr> get_outputs() const override {
        return m_outputs;
    }
    std::vector<Place::Ptr> get_places() const;

    // Other interface methods

private:
    void load_model_structure(const std::string& json_path);
    void load_metadata(const std::string& yml_path);
    void load_weights(const std::string& params_path);

    // Helper to collect all places for get_places()
    std::vector<Place::Ptr> collect_all_places() const;

    nlohmann::json m_model_json;
    std::map<std::string, std::shared_ptr<JSONTensorPlace>> m_var_places;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::shared_ptr<TelemetryExtension> m_telemetry;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov