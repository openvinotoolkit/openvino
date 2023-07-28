// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"
#include <typeinfo>


namespace ov {
namespace snippets {
namespace pass {
/**
 * @brief Manager is like ov::pass::Manager, but allows to insert new passes at arbitrary places in the pipeline
 * @ingroup snippets
 */
class Manager : public ov::pass::Manager {
public:
    ~Manager() override = default;
    using PassBase = ov::pass::PassBase;
    using Validate = ov::pass::Validate;
    /**
    * @brief PassPosition describes a particular position in a transformation pipeline,
     * where a new transformation should be inserted.
     * @param pass_name name of the anchor pass, the new pass will be inserted before/after it.
     * Empty pass_name could mean either beginning or the end of the pipeline depending on the `after` flag.
     * No default value.
     * @param after `true` if the new pass should be inserted before the anchor pass, `false` otherwise (default).
     * If `pass_name` is empty, `true` means the end, and `false` - the beginning of the pipeline.
     * @param pass_instance the number of the pass with matching `pass_name` to be considered as the anchor pass.
     * 0 (default) means the first pass with `pass_name` will be considered as the anchor pass.
    * @ingroup snippets
    */
    class PassPosition {
        const std::string m_pass_name;
        const size_t m_pass_instance;
        const bool m_after;

    public:
        using pass_list_type = std::vector<std::shared_ptr<ov::pass::PassBase>>;
        explicit PassPosition(std::string pass_name, bool after = false, size_t pass_instance = 0)
        : m_pass_name(std::move(pass_name)), m_pass_instance(pass_instance), m_after(after) {
            OPENVINO_ASSERT((!m_pass_name.empty()) || (m_pass_instance == 0),
                            "Non-zero pass_instance is not allowed if pass_name is not provided");
        }

        pass_list_type::const_iterator get_insert_position(const pass_list_type& pass_list) const;
    };
    using PositionedPass = std::pair<PassPosition, std::shared_ptr<PassBase>>;

    template <typename T, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        return ov::pass::Manager::register_pass<T>(args...);
    }
    template <typename T, class Pos,  class... Args, std::enable_if<std::is_same<PassPosition, Pos>::value, bool>() = true>
    std::shared_ptr<T> register_pass(const PassPosition& pass_id, Args&&... args) {
        auto rc = insert_pass<T>(pass_id, std::forward<Args>(args)...);
        rc->set_pass_config(m_pass_config);
        if (!m_pass_config->is_enabled<T>()) {
            m_pass_config->disable<T>();
        }
        return rc;
    }

    std::shared_ptr<PassBase> register_pass_instance(const PassPosition& pass_id, std::shared_ptr<PassBase> pass) {
        pass->set_pass_config(m_pass_config);
        auto insert_pos = pass_id.get_insert_position(m_pass_list);

        insert_pos = m_pass_list.insert(insert_pos, pass);
        if (m_per_pass_validation) {
            // Note: insert_pos points to the newly inserted pass, so advance to validate the pass results
            std::advance(insert_pos, 1);
            m_pass_list.insert(insert_pos, std::make_shared<ov::pass::Validate>());
        }
        return pass;
    }


protected:
    template <typename T, class... Args>
    std::shared_ptr<T> insert_pass(const PassPosition& pass_id, Args&&... args) {
        static_assert(std::is_base_of<PassBase, T>::value, "Attempt to insert pass that is not derived from PassBase");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<PassBase>(pass);

        const auto& insert_pos = pass_id.get_insert_position(m_pass_list);
        m_pass_list.insert(insert_pos, pass_base);
        if (m_per_pass_validation) {
            m_pass_list.insert(insert_pos, std::make_shared<ov::pass::Validate>());
        }

        return pass;
    }
};

} // namespace pass
} // namespace snippets
} // namespace ov
