// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "attention.hpp"


namespace ov ::npuw {
namespace function {



class FlashAttention {
public:
    enum tagFlashModel : uint8_t {
        eTile, eConcat, eDivide, eLast
    };

    std::vector<std::shared_ptr<ov::Model>> models;

public:
    size_t num_models() const {
        return models.size();
    }
    // Factory method
    static std::optional<FlashAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function

namespace compiled {
class FlashAttention {
public:
    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };
    std::vector<ov::SoPtr<ov::ICompiledModel>> _compiled_models;
    std::vector<std::shared_ptr<ov::Model>> _models_to_compile;
    std::vector<Param> params;
    std::size_t mask_idx = 0u;

public:
    explicit FlashAttention(const function::FlashAttention& func_flash_attention);


    // TODO: why do we need external compilation call
    void set_compiled_models(std::vector<ov::SoPtr<ov::ICompiledModel>> && models) {
        _compiled_models = std::move(models);

        // Clear temporary models storage
        _models_to_compile.clear();
        _models_to_compile.shrink_to_fit();
    }
};

}  // namespace compiled


// TODO: review this stuff
namespace runtime {
namespace flash_attention {

// A base class to decide the work-scope from some feature
class Selector {
public:
    enum class Case { PREFILL, GENERATE, UNKNOWN };

    using Ptr = std::shared_ptr<Selector>;
    virtual ~Selector() = default;
    virtual void prepare(int64_t past_len) = 0;
    virtual int64_t length() const = 0;
    virtual int64_t past_length() const = 0;

    // Getter for the selected pyramid model ID (updated by prepare())
    std::size_t pyramid_id() const {
        return m_pyramid_id;
    }

    Case this_case() const {
        return m_case;
    }

protected:
    Case m_case = Case::UNKNOWN;
    std::size_t m_pyramid_id = 0;  // Selected pyramid model ID, updated by prepare()
};

// No dynamic dispatch - just run over the whole range
class All final : public Selector {
    std::size_t m_pyramid_count = 0;

    public:
        explicit All(std::size_t pyramid_count) : m_pyramid_count(pyramid_count) {}

        void prepare(int64_t past_len) override {
            // Always use the largest pyramid model (last one)
            m_pyramid_id = m_pyramid_count > 0 ? m_pyramid_count - 1 : 0;
        }
        int64_t length() const override {
            return -1;
        }
        int64_t past_length() const override {
            OPENVINO_NOT_IMPLEMENTED;
        }
};

// Define work scope based on position ids
class PositionIDs final : public Selector {
    std::size_t m_position_ids_idx = 0u;
    int64_t m_current_length = 0;
    int64_t m_past_length = 0;
    std::size_t m_query_size = 0u;

    // Store pyramid attention reference for pyramid model selection
    const compiled::FlashAttention* m_pyramid_attention = nullptr;

    const ov::ISyncInferRequest& m_rq;

    PositionIDs(std::size_t param_idx, const compiled::FlashAttention& d, const ov::ISyncInferRequest& rq);
    void prepare(int64_t past_len) override;
    int64_t length() const override;
    int64_t past_length() const override;

public:
    static Selector::Ptr find(const compiled::FlashAttention& d, const ov::ISyncInferRequest& rq);
};
}  // namespace flash_attention
}  // namespace runtime


// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw::function
