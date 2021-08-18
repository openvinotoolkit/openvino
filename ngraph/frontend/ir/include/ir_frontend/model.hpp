// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include <inference_engine.hpp>
#include <ir_frontend/utility.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <pugixml.hpp>

namespace ngraph {
namespace frontend {
class IR_API InputModelIR : public InputModel {
    friend class FrontEndIR;

    pugi::xml_node m_root;
    InferenceEngine::Blob::CPtr m_weights;
    std::vector<InferenceEngine::IExtensionPtr> m_exts;

public:
    explicit InputModelIR(const pugi::xml_node& root,
                          const InferenceEngine::Blob::CPtr& weights,
                          const std::vector<InferenceEngine::IExtensionPtr>& exts)
        : m_root(root),
          m_weights(weights),
          m_exts(exts) {}

    std::shared_ptr<Function> convert();
};

}  // namespace frontend
}  // namespace ngraph
