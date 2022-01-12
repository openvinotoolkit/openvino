// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <string>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {

struct FRONTEND_API ModelAnalysisData {
    std::map<std::string, ov::PartialShape> inputs_shape_map;
    std::map<std::string, ov::element::Type> inputs_type_map;
};

class FRONTEND_API ModelAnalysisExtension : public ov::Extension {
public:
    using model_analysis_callback = std::function<void(const ModelAnalysisData& data)>;

    ModelAnalysisExtension() : m_callback{[](const ModelAnalysisData& data) {}} {}
    ModelAnalysisExtension(const model_analysis_callback& callback) : m_callback{callback} {}
    ModelAnalysisExtension(model_analysis_callback&& callback) : m_callback{std::move(callback)} {}

    void report_model_analysis(const ModelAnalysisData& data) const;

private:
    model_analysis_callback m_callback;
};
}  // namespace frontend
}  // namespace ov
