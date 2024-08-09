// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "positioned_pass.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @brief Manager is like ov::pass::Manager, but allows to insert new passes at arbitrary places in the pipeline
 * @ingroup snippets
 */
class Manager : public ov::pass::Manager {
public:
    Manager(std::shared_ptr<ov::pass::PassConfig> pass_config = std::make_shared<ov::pass::PassConfig>(),
            std::string name = "UnnamedSnippetsManager");
    ~Manager() override = default;
    using PassBase = ov::pass::PassBase;
    using Validate = ov::pass::Validate;
    using PositionedPassBase = PositionedPass<PassBase>;

    template <typename T, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        return ov::pass::Manager::register_pass<T>(args...);
    }
    template <typename T, class Pos,  class... Args, std::enable_if<std::is_same<PassPosition, Pos>::value, bool>() = true>
    std::shared_ptr<T> register_pass(const PassPosition& position, Args&&... args) {
        static_assert(std::is_base_of<PassBase, T>::value, "Attempt to insert pass that is not derived from PassBase");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto rc = insert_pass_instance(position, pass);
        rc->set_pass_config(m_pass_config);
        if (!m_pass_config->is_enabled<T>()) {
            m_pass_config->disable<T>();
        }
        return rc;
    }

    std::shared_ptr<PassBase> register_pass_instance(const PassPosition& pass_id, const std::shared_ptr<PassBase>& pass);
    void register_positioned_passes(const std::vector<PositionedPassBase>& pos_passes);

protected:
    std::shared_ptr<PassBase> insert_pass_instance(const PassPosition& position, const std::shared_ptr<PassBase>& pass);
};

} // namespace pass
} // namespace snippets
} // namespace ov
