// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <set>

namespace ov {
namespace util {
/// \brief Get the names environment variable as a string.
/// \param env_var The string name of the environment variable to get.
/// \return Returns string by value or an empty string if the environment
///         variable is not set.
std::string getenv_string(const char* env_var);

/// \brief Get the names environment variable as an integer. If the value is not a
///        valid integer then an exception is thrown.
/// \param env_var The string name of the environment variable to get.
/// \param default_value The value to return if the environment variable is not set.
/// \return Returns value or default_value if the environment variable is not set.
int32_t getenv_int(const char* env_var, int32_t default_value = -1);

/// \brief Get the names environment variable as a boolean. If the value is not a
///        valid boolean then an exception is thrown. Valid booleans are one of
///        1, 0, on, off, true, false
///        All values are case insensitive.
///        If the environment variable is not set the default_value is returned.
/// \param env_var The string name of the environment variable to get.
/// \param default_value The value to return if the environment variable is not set.
/// \return Returns the boolean value of the environment variable.
bool getenv_bool(const char* env_var, bool default_value = false);

/// \brief Get the tokens as a boolean from the environment variable.
///        That is true if the environment variable contains any of the enabling tokens
///        and does not contain any of the disabling tokens, otherwise that is false.
///        For example: "enabling_2" -> true; "enabling_1, disabling_1" -> false
///        If the environment variable is not set the default_value is returned.
///        The environment variable value is treated as a case insensitive string
///        containing comma separated tokens (spaces are trimmed from both ends of a token).
/// \param env_var The string name of the environment variable to get.
/// \param enablingTokens  The set of the lowercase string names of the enabling tokens.
/// \param disablingTokens The set of the lowercase string names of the disabling tokens.
/// \param default_value The value to return if the environment variable is not set.
/// \return Returns the boolean value of the tokens from the environment variable.
bool getenv_tokens_bool(const char* env_var,
                        const std::set<std::string>& enablingTokens,
                        const std::set<std::string>& disablingTokens,
                        bool default_value = false);

/// \brief Get the token as a boolean from the OV_ENABLE environment variable.
///        That is true if the token is enabled, and false otherwise.
///        For example: "all", "snippets" -> true; "all,-snippets", "ALL, -SnIpPeTs" -> false
///        If the token is not supported then an exception is thrown.
///        Supported tokens are: "all", snippets"
/// \param token The lowercase string name of the token to get.
/// \return Returns the boolean value of the token from the OV_ENABLE environment variable.
bool getenv_ov_enable_bool(const std::string& token);

/// \brief Get the token as a boolean from the OV_DUMP_IR environment variable.
///        That is true if the token is enabled, and false otherwise.
///        For example: "all", "snippets" -> true; "all,-snippets", "ALL, -SnIpPeTs" -> false
///        If the token is not supported then an exception is thrown.
///        Supported tokens are: "all", "snippets"
/// \param token The lowercase string name of the token to get.
/// \return Returns the boolean value of the token from the OV_DUMP_IR environment variable.
bool getenv_ov_dump_ir_bool(const std::string& token);
}  // namespace util
}  // namespace ov
