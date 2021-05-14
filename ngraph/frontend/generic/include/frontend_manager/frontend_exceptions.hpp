// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "ngraph/check.hpp"

namespace ngraph {
namespace frontend {

enum class FrontEndErrorCode {
    GENERAL_ERROR,
    NOT_IMPLEMENTED,
    OP_VALIDATION_FAILED,
    NGRAPH_NODE_CREATION_FAILED,
    INITIALIZATION_ERROR,
};

/// \brief Base class for check failure exceptions.
class CheckFailureFrontEnd : public CheckFailure {
public:

    CheckFailureFrontEnd(FrontEndErrorCode error_code,
                         const CheckLocInfo &check_loc_info,
                         const std::string &context,
                         const std::string &explanation)
            : CheckFailure(check_loc_info, "FrontEnd API failed" + context, explanation),
              m_error_code(error_code) {
    }

    FrontEndErrorCode getErrorCode() const { return m_error_code; }
private:
    FrontEndErrorCode m_error_code;
};

#define FRONT_END_CHECK_HELPER2(error_code, exc_class, ctx, check, ...)                            \
    do                                                                                             \
    {                                                                                              \
        if (!(check))                                                                              \
        {                                                                                          \
            ::std::stringstream ss___;                                                             \
            ::ngraph::write_all_to_stream(ss___, __VA_ARGS__);                                     \
            throw exc_class((error_code),                                                          \
                (::ngraph::CheckLocInfo{__FILE__, __LINE__, #check}), (ctx), ss___.str());         \
        }                                                                                          \
    } while (0)

#define FRONT_END_CHECK_HELPER1(error_code, exc_class, ctx, check)                                 \
    do                                                                                             \
    {                                                                                              \
        if (!(check))                                                                              \
        {                                                                                          \
            throw exc_class((error_code),                                                          \
                (::ngraph::CheckLocInfo{__FILE__, __LINE__, #check}), (ctx), "");                  \
        }                                                                                          \
    } while (0)

#define FRONT_END_CALL_OVERLOAD(name, error_code, exc_class, ctx, ...)                             \
    GLUE(OVERLOAD_MACRO(name, COUNT_ARGS_MAXN(__VA_ARGS__)),                                       \
        (error_code, exc_class, ctx, __VA_ARGS__))

#define FRONT_END_CHECK_HELPER(error_code, exc_class, ctx, ...)                                    \
    FRONT_END_CALL_OVERLOAD(FRONT_END_CHECK_HELPER, error_code, exc_class, ctx, __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param error_code Additional indicator of the type of error.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailurePDPD if `cond` is false.
#define FRONT_END_CHECK(error_code, ...) FRONT_END_CHECK_HELPER(error_code, ::ngraph::frontend::CheckFailureFrontEnd, "", __VA_ARGS__)
#define FRONT_END_NOT_IMPLEMENTED(NAME) FRONT_END_CHECK(FrontEndErrorCode::NOT_IMPLEMENTED, false, #NAME" is not implemented for this FrontEnd class")

} // frontend
} // ngraph