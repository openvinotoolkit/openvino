// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "npuw/compiled_model.hpp"

namespace ov {
namespace npuw {

class KokoroInferRequest;

struct KokoroSplitResult {
    std::shared_ptr<ov::Model> model_a;
    std::shared_ptr<ov::Model> model_b;
};

struct KokoroConfig {
    uint64_t block_size = 200;
    uint64_t overlap_size = 20;
};

class KokoroCompiledModel : public ov::npuw::ICompiledModel {
public:
    KokoroCompiledModel(const std::shared_ptr<ov::Model>& model,
                        const std::shared_ptr<const ov::IPlugin>& plugin,
                        const ov::AnyMap& properties);

    void export_model(std::ostream& model) const override;
    static std::shared_ptr<KokoroCompiledModel> import_model(std::istream& stream,
                                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                                             const ov::AnyMap& properties);

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    uint64_t block_size() const noexcept {
        return m_kokoro_cfg.block_size;
    }

    uint64_t overlap_size() const noexcept {
        return m_kokoro_cfg.overlap_size;
    }

    ov::SoPtr<ov::ICompiledModel> model_a() const {
        return m_model_a_compiled;
    }
    std::shared_ptr<ov::npuw::ICompiledModel> model_b() const {
        return m_model_b_compiled;
    }

private:
    KokoroConfig m_kokoro_cfg;
    std::string m_name;
    std::shared_ptr<::intel_npu::OptionsDesc> m_options_desc;
    ::intel_npu::Config m_cfg;

    std::shared_ptr<ov::ISyncInferRequest> create_kokoro_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    ov::SoPtr<ov::ICompiledModel> m_model_a_compiled;
    std::shared_ptr<ov::npuw::ICompiledModel> m_model_b_compiled;
};

}  // namespace npuw
}  // namespace ov
