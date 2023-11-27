// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/pass/pass_position.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
class PassPipeline {
public:
    PassPipeline() = default;

    void run(lowered::LinearIR& linear_ir) const;
    void register_pass(const std::shared_ptr<Pass>& pass);

    template<typename T, class... Args>
    void register_pass(Args&&... args) {
        static_assert(std::is_base_of<Pass, T>::value, "Pass not derived from lowered::Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(pass);
    }

    struct PositionedPass {
        ov::snippets::pass::PassPosition position;
        std::shared_ptr<Pass> pass;
        PositionedPass(ov::snippets::pass::PassPosition arg_pos, std::shared_ptr<Pass> arg_pass)
        : position(std::move(arg_pos)), pass(std::move(arg_pass)) {
        }
    };

    template <typename T, class Pos,  class... Args, std::enable_if<std::is_same<ov::snippets::pass::PassPosition, Pos>::value, bool>() = true>
    void register_pass(const ov::snippets::pass::PassPosition& position, Args&&... args) {
        static_assert(std::is_base_of<Pass, T>::value, "Attempt to insert pass that is not derived from Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        insert_pass_instance(position, pass);
    }

    void register_positioned_passes(const std::vector<PositionedPass>& pos_passes);

protected:
    void insert_pass_instance(const ov::snippets::pass::PassPosition& position, const std::shared_ptr<Pass>& pass);

private:
    std::vector<std::shared_ptr<Pass>> m_passes;
};

class SubgraphPassPipeline {
public:
    SubgraphPassPipeline() = default;

    void run(const lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) const;
    const std::vector<std::shared_ptr<SubgraphPass>>& get_passes() const;
    void register_pass(const std::shared_ptr<SubgraphPass>& pass);
    bool empty() const { return m_passes.empty(); }

    template<typename T, class... Args>
    void register_pass(Args&&... args) {
        static_assert(std::is_base_of<SubgraphPass, T>::value, "Pass not derived from lowered::SubgraphPass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(pass);
    }

    struct PositionedPass {
        ov::snippets::pass::PassPosition position;
        std::shared_ptr<SubgraphPass> pass;
        PositionedPass(ov::snippets::pass::PassPosition arg_pos, std::shared_ptr<SubgraphPass> arg_pass)
        : position(std::move(arg_pos)), pass(std::move(arg_pass)) {
        }
    };

    template <typename T, class Pos,  class... Args, std::enable_if<std::is_same<ov::snippets::pass::PassPosition, Pos>::value, bool>() = true>
    void register_pass(const ov::snippets::pass::PassPosition& position, Args&&... args) {
        static_assert(std::is_base_of<SubgraphPass, T>::value, "Attempt to insert pass that is not derived from SubgraphPass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        insert_pass_instance(position, pass);
    }

    void register_positioned_passes(const std::vector<PositionedPass>& pos_passes);

protected:
    void insert_pass_instance(const ov::snippets::pass::PassPosition& position, const std::shared_ptr<SubgraphPass>& pass);

private:
    std::vector<std::shared_ptr<SubgraphPass>> m_passes;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
