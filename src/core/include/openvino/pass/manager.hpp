// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <typeinfo>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"

namespace ov {
namespace pass {
/**
 * @brief Manager class allows to manage transformation passes
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API Manager {
public:
    Manager();
    ~Manager();

    //// \brief Construct Manager with shared PassConfig instance
    explicit Manager(std::shared_ptr<PassConfig> pass_config);

    /// \brief Register given transformation class type to execution list
    /// Example below show the basic usage of pass::Manager
    ///
    ///     pass::Manager manager;
    ///     manager.register_pass<MyTransformation>(/*transformation constructor ars*/);
    ///     manager.run_passes(f);
    ///
    /// For some purposes transformation can be registered and disabled by default.
    ///
    ///     manager.register_pass<MyTransformation, false>();
    ///
    /// \return shared_ptr to the transformation instance
    template <typename T, bool Enable = true, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        auto rc = push_pass<T>(std::forward<Args>(args)...);
        rc->set_pass_config(m_pass_config);
        if (m_per_pass_validation) {
            push_pass<Validate>();
        }
        if (!Enable && !m_pass_config->is_enabled<T>()) {
            m_pass_config->disable<T>();
        }
        return rc;
    }

    std::shared_ptr<PassBase> register_pass_instance(std::shared_ptr<PassBase> pass) {
        pass->set_pass_config(m_pass_config);
        m_pass_list.push_back(pass);
        if (m_per_pass_validation) {
            push_pass<Validate>();
        }
        return pass;
    }

    void run_passes(std::shared_ptr<Model>);

    void set_pass_visualization(bool new_state) {
        m_visualize = new_state;
    }
    /// \brief Set flag to enable/disable running Validate pass after executing
    /// each registered pass
    /// \param new_state Value "true" enables Validate pass run; "false", otherwise
    void set_per_pass_validation(bool new_state);

    /// \brief Callback is a lambda function that can be used by registered transformations.
    /// The main purpose of this callback is to provide a way for plugins to disable/enable
    /// transformations based on some conditions. In some cases plugins may want not to
    /// execute some
    /// transformations.
    /// For example plugin can disable unpleasant decompositions because of performance
    /// reasons for
    /// some cases.
    /// Callback example:
    /// auto callback = [](const std::shared_ptr<const ov::Node> & node) -> bool {
    ///     return std::dynamic_pointer_cast<const ov::opset3::DepthToSpace>(node) !=
    ///     nullptr;
    /// };
    /// This callback returns true in case of DepthToSpace operation. So when execution
    /// DepthToSpace
    /// decomposition pass will check is this decomposition needed or plugin can execute
    /// this
    /// operation directly. And of course on transformation side we need to have a response
    /// for this
    /// callback.
    /// if (transformation_callback(batch_to_space)) {
    ///     return false;
    /// }
    /// \param callback lamda function that returns true in case if node is supported by
    /// plugin and
    /// transformation is not needed
    OPENVINO_DEPRECATED("Please use get_pass_config() to configure transformation pipeline")
    void set_callback(const param_callback& callback) {
        m_pass_config->set_callback(callback);
    }
    /// \return PassConfig shared object. This object is used for transformations pipeline
    /// configuration.
    /// This object allows to disable/enable transformations execution, set callback to
    /// particular
    /// transformation. For mo details see PassConfig class.
    std::shared_ptr<PassConfig> get_pass_config() {
        return m_pass_config;
    }

protected:
    template <typename T, class... Args>
    std::shared_ptr<T> push_pass(Args&&... args) {
        static_assert(std::is_base_of<pass::PassBase, T>::value, "pass not derived from pass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<PassBase>(pass);
        m_pass_list.push_back(pass_base);
        return pass;
    }

    std::shared_ptr<PassConfig> m_pass_config;
    std::vector<std::shared_ptr<PassBase>> m_pass_list;
    bool m_visualize = false;
    bool m_per_pass_validation = true;
};
}  // namespace pass
}  // namespace ov
