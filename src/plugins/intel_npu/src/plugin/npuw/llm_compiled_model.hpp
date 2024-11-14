// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "compiled_model.hpp"

namespace ov {
namespace npuw {

class LLMCompiledModel : public ov::npuw::ICompiledModel {
public:
    LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const ov::AnyMap& properties);
    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    // FIXME: Publicly available for LLMInferRequest
    std::shared_ptr<ov::npuw::CompiledModel> kvcache_compiled;
    std::shared_ptr<ov::npuw::CompiledModel> prefill_compiled;

private:
    std::shared_ptr<ov::Model> orig_model;
    std::shared_ptr<ov::ISyncInferRequest> create_llm_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
};

} // namespace npuw
} // namespace ov
