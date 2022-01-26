// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/manager.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace test {

inline std::shared_ptr<ov::Model> readModel(const std::string& model_path, const std::string& weights_path) {
    static ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{model_path};
    if (!weights_path.empty())
        params.emplace_back(weights_path);

    FE = manager.load_by_model(params);
    if (FE)
        inputModel = FE->load(params);

    if (inputModel)
        return FE->convert(inputModel);

    OPENVINO_ASSERT(false, "Failed to read the model");
}

inline std::shared_ptr<ov::Model> readModel(const std::string& model) {
    static ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;

    ov::AnyVector params{&modelStream};

    FE = manager.load_by_model(params);
    if (FE)
        inputModel = FE->load(params);

    if (inputModel)
        return FE->convert(inputModel);

    return nullptr;
}

}  // namespace test
}  // namespace ov
