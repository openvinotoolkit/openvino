// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/decoder_transformation.hpp"

#include <utility>

using namespace ov;
using namespace ov::frontend;

/// \brief Helper class to register user function as a FunctionPass
class CustomModelPass : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("frontend::CustomModelPass");
    explicit CustomModelPass(std::function<bool(std::shared_ptr<ov::Model>)> pass) : m_pass(std::move(pass)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        return m_pass(f);
    }

private:
    std::function<bool(std::shared_ptr<ov::Model>)> m_pass;
};

/// \brief Helper class to register user matcher pass initialization as a MatcherPass
class CustomMatcherPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("frontend::CustomMatcherPass");
    explicit CustomMatcherPass(const std::function<void(ov::pass::MatcherPass*)>& matcher_pass_initializer) {
        matcher_pass_initializer(this);
    }
};

DecoderTransformationExtension::DecoderTransformationExtension(
    const std::function<bool(const std::shared_ptr<ov::Model>)>& function_pass)
    : m_registration([=](ov::pass::Manager& manager) {
          manager.register_pass<CustomModelPass>(function_pass);
      }) {}

DecoderTransformationExtension::DecoderTransformationExtension(
    const std::function<void(ov::pass::MatcherPass*)>& matcher_pass_initializer)
    : m_registration([=](ov::pass::Manager& manager) {
          manager.register_pass<CustomMatcherPass>(matcher_pass_initializer);
      }) {}

void DecoderTransformationExtension::register_pass(ov::pass::Manager& manager) const {
    if (m_registration) {
        m_registration(manager);
    }
}
