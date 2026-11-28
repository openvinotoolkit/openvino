// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "intel_npu/config/config.hpp"
#include "npuw/compiled_model.hpp"
#include "openvino/runtime/icore.hpp"

namespace ov {
namespace npuw {

// Runs a dynamic-shape BAGEL submodel through static NPUW variants compiled on first use.
//
// The MoT language model (detected by its und_index/gen_index inputs) is padded to sequence and
// KV-length buckets: padded KV slots and rows are masked, calls without generation tokens get one
// masked dummy generation token, and outputs are cropped back to the real shapes. Any other model
// (vision encoder, VAE) gets one static variant per distinct input-shape signature.
class BagelCompiledModel : public ov::npuw::ICompiledModel {
public:
    enum class Mode { LANGUAGE_MODEL, PER_SHAPE };

    // Language model variant: padded sequence length, generation token count, padded KV length
    struct LMKey {
        std::size_t seq;
        std::size_t gen;
        std::size_t kv;
        bool operator<(const LMKey& other) const {
            return std::tie(seq, gen, kv) < std::tie(other.seq, other.gen, other.kv);
        }
    };
    using ShapeKey = std::vector<ov::Shape>;

    BagelCompiledModel(const std::shared_ptr<ov::Model>& model,
                       const std::shared_ptr<const ov::IPlugin>& plugin,
                       const ov::AnyMap& properties);

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    Mode mode() const noexcept {
        return m_mode;
    }
    std::size_t seq_bucket(std::size_t n) const;
    std::size_t kv_bucket(std::size_t n) const;
    // Variants with several generation tokens only see full-attention masks, whose rows are all equal
    bool folds_mask(std::size_t n_gen) const noexcept {
        return m_fold_mask && n_gen > 1;
    }

    ov::SoPtr<ov::ICompiledModel> lm_variant(const LMKey& key) const;
    ov::SoPtr<ov::ICompiledModel> shape_variant(const ShapeKey& key) const;

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    ov::SoPtr<ov::ICompiledModel> compile_variant(const std::map<std::size_t, ov::PartialShape>& shapes,
                                                  const std::string& tag,
                                                  bool drop_logits,
                                                  bool fold_mask) const;

    std::shared_ptr<ov::Model> m_model;
    // Variants are compiled after compile_model() returns, when the user may have released the Core
    std::shared_ptr<ov::ICore> m_core;
    Mode m_mode;
    ov::AnyMap m_inner_props;
    std::vector<std::size_t> m_seq_buckets;
    std::vector<std::size_t> m_kv_buckets;
    bool m_fold_mask = false;

    std::shared_ptr<::intel_npu::OptionsDesc> m_options_desc;
    ::intel_npu::Config m_cfg;

    mutable std::mutex m_mutex;
    mutable std::map<LMKey, ov::SoPtr<ov::ICompiledModel>> m_lm_variants;
    mutable std::map<ShapeKey, ov::SoPtr<ov::ICompiledModel>> m_shape_variants;
};

}  // namespace npuw
}  // namespace ov
