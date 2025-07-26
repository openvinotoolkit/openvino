// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass_config.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface PassBase
 * @brief Base class for transformations on linear IR
 * @ingroup snippets
 */
class PassBase : public std::enable_shared_from_this<PassBase> {
public:
    PassBase() = default;
    virtual ~PassBase() = default;
    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static{"snippets::lowered::pass::PassBase"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }

    /**
     * @brief Merges the current pass with other (e.g. during 2 pass pipelines fusion).
     * @param other  Pointer on the another pass.
     * @return The merged pass
     * @attention If 'other' pass is empty (aka nullptr), it can be merged to any other pass.
     * @attention If the merge fails, then nullptr is returned.
     */
    virtual std::shared_ptr<PassBase> merge([[maybe_unused]] const std::shared_ptr<PassBase>& other) {
        return nullptr;
    }
};

/**
 * @interface Pass
 * @brief Base class for LIR passes which are performed on a full LIR body
 * @ingroup snippets
 */
class Pass : public PassBase {
public:
    OPENVINO_RTTI("snippets::lowered::pass::Pass")
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    virtual bool run(lowered::LinearIR& linear_ir) = 0;
};

/**
 * @interface ConstPass
 * @brief Base class for LIR passes which are performed on a full LIR body but doesn't change it
 * @ingroup snippets
 */
class ConstPass : public PassBase {
public:
    OPENVINO_RTTI("snippets::lowered::pass::ConstPass")
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    virtual bool run(const lowered::LinearIR& linear_ir) = 0;
};

/**
 * @interface RangedPass
 * @brief Base class for LIR passes which are performed on a range of a LIR body
 * @ingroup snippets
 */
class RangedPass : public PassBase {
public:
    OPENVINO_RTTI("snippets::lowered::pass::RangedPass")
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @param begin begin of the range on which the pass is performed
     * @param end end of the range on which the pass is performed
     * @return status of the pass
     */
    virtual bool run(lowered::LinearIR& linear_ir,
                     lowered::LinearIR::constExprIt begin,
                     lowered::LinearIR::constExprIt end) = 0;
};

class PassPipeline {
public:
    using PositionedPassLowered = snippets::pass::PositionedPass<lowered::pass::PassBase>;

    PassPipeline();
    explicit PassPipeline(const std::shared_ptr<PassConfig>& pass_config);

    [[nodiscard]] const std::vector<std::shared_ptr<PassBase>>& get_passes() const {
        return m_passes;
    }
    [[nodiscard]] const std::shared_ptr<PassConfig>& get_pass_config() const {
        return m_pass_config;
    }
    [[nodiscard]] bool empty() const {
        return m_passes.empty();
    }

    void register_pass(const snippets::pass::PassPosition& position, const std::shared_ptr<PassBase>& pass);
    void register_pass(const std::shared_ptr<PassBase>& pass);

    template <typename T, class... Args>
    void register_pass(Args&&... args) {
        static_assert(std::is_base_of<PassBase, T>::value, "Pass not derived from lowered::Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(pass);
    }
    template <typename T,
              class Pos,
              class... Args,
              std::enable_if<std::is_same_v<snippets::pass::PassPosition, Pos>, bool>() = true>
    void register_pass(const snippets::pass::PassPosition& position, Args&&... args) {
        static_assert(std::is_base_of<PassBase, T>::value, "Pass not derived from lowered::Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(position, pass);
    }

    void register_positioned_passes(const std::vector<PositionedPassLowered>& pos_passes);

    void run(lowered::LinearIR& linear_ir) const;
    void run(const lowered::LinearIR& linear_ir) const;
    void run(lowered::LinearIR& linear_ir,
             lowered::LinearIR::constExprIt begin,
             lowered::LinearIR::constExprIt end) const;

    /**
     * @brief Merges 2 pass pipelines into one
     * @param lhs first pass pipeline
     * @param rhs second pass pipeline
     * @return the merged pass pipeline
     * @attention the function can not be used in case when one of the pipelines contains passes whose running order is
     * important.
     */
    static PassPipeline merge_pipelines(const PassPipeline& lhs, const PassPipeline& rhs);

private:
    std::shared_ptr<PassConfig> m_pass_config;
    std::vector<std::shared_ptr<PassBase>> m_passes;
};

}  // namespace ov::snippets::lowered::pass
