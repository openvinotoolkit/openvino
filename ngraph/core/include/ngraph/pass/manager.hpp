//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <list>
#include <memory>
#include <typeinfo>
#include <vector>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/pass/validate.hpp"

namespace ngraph
{
    namespace pass
    {
        class Manager;
    }
}

class NGRAPH_API ngraph::pass::Manager
{
public:
    Manager();
    ~Manager();

    template <typename T, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args)
    {
        auto rc = push_pass<T>(std::forward<Args>(args)...);
        if (m_per_pass_validation)
        {
            push_pass<Validate>();
        }
        return rc;
    }

    void run_passes(std::shared_ptr<Function>);

    void set_pass_visualization(bool new_state) { m_visualize = new_state; }
    /// \brief Set flag to enable/disable running Validate pass after executing
    /// each registered pass
    /// \param new_state Value "true" enables Validate pass run; "false", otherwise
    void set_per_pass_validation(bool new_state) { m_per_pass_validation = new_state; }
    /// \brief Callback is a lambda function that can be used by registered transformations.
    /// The main purpose of this callback is to provide a way for plugins to disable/enable
    /// transformations. In some cases plugins may want not to execute some transformations.
    /// For example plugin can disable unpleasant decompositions because of performance reasons.
    /// Callback example:
    /// auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
    ///     return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
    /// };
    /// This callback returns true in case of DepthToSpace operation. So when execution DepthToSpace
    /// decomposition pass will check is this decomposition needed or plugin can execute this
    /// operation directly. And of course on transformation side we need to have a response for this
    /// callback.
    /// if (m_transformation_callback(batch_to_space)) {
    ///     return false;
    /// }
    /// \param callback lamda function that returns true in case if node is supported by plugin and
    /// transformation is not needed
    void set_callback(param_callback callback)
    {
        m_transformation_callback = callback;
        m_has_default_callback = false;
    }

protected:
    template <typename T, class... Args>
    std::shared_ptr<T> push_pass(Args&&... args)
    {
        static_assert(std::is_base_of<pass::PassBase, T>::value, "pass not derived from pass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<PassBase>(pass);
        m_pass_list.push_back(pass_base);
        return pass;
    }

    param_callback m_transformation_callback = [](const std::shared_ptr<const Node>&) -> bool {
        return false;
    };
    bool m_has_default_callback = true;

    std::vector<std::shared_ptr<PassBase>> m_pass_list;
    bool m_visualize = false;
    bool m_per_pass_validation = true;
};
