//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <string>

#include <ngraph/ngraph_visibility.hpp>

namespace ngraph
{
    /// \brief Get the names environment variable as a string.
    /// \param env_var The string name of the environment variable to get.
    /// \return Returns string by value or an empty string if the environment
    ///         variable is not set.
    NGRAPH_API
    std::string getenv_string(const char* env_var);

    /// \brief Get the names environment variable as an integer. If the value is not a
    ///        valid integer then an exception is thrown.
    /// \param env_var The string name of the environment variable to get.
    /// \param default_value The value to return if the environment variable is not set.
    /// \return Returns value or default_value if the environment variable is not set.
    NGRAPH_API
    int32_t getenv_int(const char* env_var, int32_t default_value = -1);

    /// \brief Get the names environment variable as a boolean. If the value is not a
    ///        valid boolean then an exception is thrown. Valid booleans are one of
    ///        1, 0, on, off, true, false
    ///        All values are case insensitive.
    ///        If the environment variable is not set the default_value is returned.
    /// \param env_var The string name of the environment variable to get.
    /// \param default_value The value to return if the environment variable is not set.
    /// \return Returns the boolean value of the environment variable.
    NGRAPH_API
    bool getenv_bool(const char* env_var, bool default_value = false);
}
