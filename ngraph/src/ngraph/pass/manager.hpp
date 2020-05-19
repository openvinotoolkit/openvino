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

#include "ngraph/pass/manager_state.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/pass/validate.hpp"

namespace ngraph
{
    namespace pass
    {
        class Manager;
        class ManagerState;
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

    void run_passes(std::shared_ptr<Function>, bool transitive = true);

    ManagerState& get_state();
    PassConfig& get_pass_config() { return m_pass_config; }
    void set_pass_config(const PassConfig& pass_config) { m_pass_config = pass_config; }
    void set_pass_visualization(bool new_state) { m_visualize = new_state; }
    void set_pass_serialization(bool new_state) { m_serialize = new_state; }
    /// \brief Set flag to enable/disable running Validate pass after executing
    /// each registered pass
    /// \param new_state Value "true" enables Validate pass run; "false", otherwise
    void set_per_pass_validation(bool new_state) { m_per_pass_validation = new_state; }
private:
    template <typename T, class... Args>
    std::shared_ptr<T> push_pass(Args&&... args)
    {
        static_assert(std::is_base_of<pass::PassBase, T>::value, "pass not derived from pass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<PassBase>(pass);
        m_pass_list.push_back(pass_base);
        if (m_visualize || m_serialize)
        {
#ifdef _WIN32
            // MSVC produce a human-readable type name like class ngraph::pass::LikeReplacement
            // by typeid(T).name(). Later ofstream doesn't accept it as a valid file name.
            //
            std::string str = typeid(T).name();
            auto pos = str.find_last_of(":");
            m_pass_names.push_back(str.substr(pos + 1));
#elif defined(__linux) || defined(__APPLE__)
            m_pass_names.push_back(typeid(T).name());
#endif
        }
        return pass;
    }

    std::vector<std::string> m_pass_names;
    std::vector<std::shared_ptr<PassBase>> m_pass_list;
    ManagerState m_state;
    PassConfig m_pass_config;
    bool m_visualize = false;
    bool m_serialize = false;
    bool m_per_pass_validation = true;
};
