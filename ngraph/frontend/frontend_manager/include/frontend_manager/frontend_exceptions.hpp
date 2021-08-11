// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "frontend_manager_defs.hpp"
#include "ngraph/check.hpp"

namespace ov
{
    namespace frontend
    {
        class FRONTEND_API GeneralFailure : public CheckFailure
        {
        public:
            GeneralFailure(const CheckLocInfo& check_loc_info,
                           const std::string& context,
                           const std::string& explanation)
                : CheckFailure(check_loc_info,
                               "FrontEnd API failed with GeneralFailure: " + context,
                               explanation)
            {
            }
        };

        class FRONTEND_API InitializationFailure : public CheckFailure
        {
        public:
            InitializationFailure(const CheckLocInfo& check_loc_info,
                                  const std::string& context,
                                  const std::string& explanation)
                : CheckFailure(check_loc_info,
                               "FrontEnd API failed with InitializationFailure: " + context,
                               explanation)
            {
            }
        };

        class FRONTEND_API OpValidationFailure : public CheckFailure
        {
        public:
            OpValidationFailure(const CheckLocInfo& check_loc_info,
                                const std::string& context,
                                const std::string& explanation)
                : CheckFailure(check_loc_info,
                               "FrontEnd API failed with OpValidationFailure: " + context,
                               explanation)
            {
            }
        };

        class FRONTEND_API OpConversionFailure : public CheckFailure
        {
        public:
            OpConversionFailure(const CheckLocInfo& check_loc_info,
                                const std::string& context,
                                const std::string& explanation)
                : CheckFailure(check_loc_info,
                               "FrontEnd API failed with OpConversionFailure: " + context,
                               explanation)
            {
            }
        };

        class FRONTEND_API NotImplementedFailure : public CheckFailure
        {
        public:
            NotImplementedFailure(const CheckLocInfo& check_loc_info,
                                  const std::string& context,
                                  const std::string& explanation)
                : CheckFailure(check_loc_info,
                               "FrontEnd API failed with NotImplementedFailure: " + context,
                               explanation)
            {
            }
        };

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::GeneralFailure if `cond` is false.
#define FRONT_END_GENERAL_CHECK(...)                                                               \
    NGRAPH_CHECK_HELPER(::ov::frontend::GeneralFailure, "", __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::InitializationFailure if `cond` is false.
#define FRONT_END_INITIALIZATION_CHECK(...)                                                        \
    NGRAPH_CHECK_HELPER(::ov::frontend::InitializationFailure, "", __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::OpConversionFailure if `cond` is false.
#define FRONT_END_OP_CONVERSION_CHECK(...)                                                         \
    NGRAPH_CHECK_HELPER(::ov::frontend::OpConversionFailure, "", __VA_ARGS__)

/// \brief Assert macro.
/// \param NAME Name of the function that is not implemented
/// \throws ::ov::frontend::NotImplementedFailure
#define FRONT_END_NOT_IMPLEMENTED(NAME)                                                            \
    NGRAPH_CHECK_HELPER(::ov::frontend::NotImplementedFailure,                                     \
                        "",                                                                        \
                        false,                                                                     \
                        #NAME " is not implemented for this FrontEnd class")

/// \brief Assert macro.
/// \param MSG Error message
/// \throws ::ov::frontend::GeneralFailure
#define FRONT_END_THROW(MSG) FRONT_END_GENERAL_CHECK(false, MSG)

    } // namespace frontend
} // namespace ov
