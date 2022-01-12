// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/model_analysis_extension.hpp"

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
void ModelAnalysisExtension::report_model_analysis(const ModelAnalysisData& data) const {
    std::cout << "------: " << data.inputs_shape_map.begin()->second << "\n";
    m_callback(data);
}
}  // namespace frontend
}  // namespace ov
