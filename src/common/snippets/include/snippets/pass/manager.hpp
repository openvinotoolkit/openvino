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
     *       where a new transformation should be inserted.
     * @param pass_name name of the anchor pass, the new pass will be inserted before/after it.
     *        Empty pass_name could mean either beginning or the end of the pipeline depending on the `after` flag.
     *        No default value. Note that pass names namespaces are not supported, ov::PassName and snippets::PassName
     *        are considered identical.
     * @param after `true` if the new pass should be inserted before the anchor pass, `false` otherwise (default).
     *        If `pass_name` is empty, `true` means the end, and `false` - the beginning of the pipeline.
     * @param pass_instance the number of the pass with matching `pass_name` to be considered as the anchor pass.
     *        0 (default) means the first pass with `pass_name` will be considered as the anchor pass.
    * @ingroup snippets
    */
    class PassPosition {
    public:
        enum class Place {Before, After, PipelineStart, PipelineEnd};
        using PassListType = std::vector<std::shared_ptr<ov::pass::PassBase>>;
        explicit PassPosition(Place pass_place);
        explicit PassPosition(Place pass_place, std::string pass_name, size_t pass_instance = 0);
        PassListType::const_iterator get_insert_position(const PassListType& pass_list) const;
    private:
        const std::string m_pass_name;
        const size_t m_pass_instance{0};
        const Place m_place{Place::Before};
    };
    struct PositionedPass {
        PassPosition position;
        std::shared_ptr<PassBase> pass;
        PositionedPass(PassPosition arg_pos, std::shared_ptr<PassBase> arg_pass)
        : position(std::move(arg_pos)), pass(std::move(arg_pass)) {
        }
    };

    template <typename T, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        return ov::pass::Manager::register_pass<T>(args...);
    }
    template <typename T, class Pos,  class... Args, std::enable_if<std::is_same<PassPosition, Pos>::value, bool>() = true>
    std::shared_ptr<T> register_pass(const PassPosition& position, Args&&... args) {
        static_assert(std::is_base_of<PassBase, T>::value, "Attempt to insert pass that is not derived from PassBase");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto rc =  insert_pass_instance(position, pass);
        rc->set_pass_config(m_pass_config);
        if (!m_pass_config->is_enabled<T>()) {
            m_pass_config->disable<T>();
        }
        return rc;
    }

    std::shared_ptr<PassBase> register_pass_instance(const PassPosition& pass_id, const std::shared_ptr<PassBase>& pass);
    void register_positioned_passes(const std::vector<PositionedPass>& pos_passes);

protected:
    std::shared_ptr<Manager::PassBase> insert_pass_instance(const PassPosition& position, const std::shared_ptr<PassBase>& pass);
};

} // namespace pass
} // namespace snippets
} // namespace ov
