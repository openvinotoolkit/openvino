/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "api/C/cldnn.h"
#include "api/CPP/cldnn_defs.h"

#include <functional>
#include <stdexcept>

#define API_CAST(api_type, impl_type) \
inline api_type api_cast(impl_type* value) { return reinterpret_cast<api_type>(value); } \
inline impl_type* api_cast(api_type value) { return reinterpret_cast<impl_type*>(value); }

namespace cldnn {
    struct last_err {
        /// @breif Sets the message of last error
        void set_last_error_message(const std::string& msg)
        {
            _msg = msg;
        }

        void set_last_exception(const std::exception& ex)
        {
            _msg = ex.what();
        }

        /// @breif Gets the message of last error
        const std::string& get_last_error_message()
        {
            return _msg;
        }
        static last_err& instance();
    private:
        std::string _msg;
        last_err() :_msg("Operation succeed") {}
    };

    // float <--> half convertors
    float half_to_float(uint16_t value);
    uint16_t float_to_half(float value);
}


template<typename T>
T exception_handler(cldnn_status default_error, cldnn_status* status, const T& default_result, std::function<T()> func)
{
    //NOTE for implementer: status should not be modified after successful func() call
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return func();
    }
    catch (const cldnn::error& err)
    {
        if (status)
            *status = err.status();
        cldnn::last_err::instance().set_last_exception(err);

#ifndef NDEBUG
        static_cast<void>(default_result);
        throw;
#endif
    }
    catch (const std::exception& err)
    {
        if (status)
            *status = default_error;
        cldnn::last_err::instance().set_last_exception(err);

#ifndef NDEBUG
        static_cast<void>(default_result);
        throw;
#endif
    }
    catch (...)
    {
        if (status)
            *status = default_error;
        cldnn::last_err::instance().set_last_error_message("error unknown");

#ifndef NDEBUG
        static_cast<void>(default_result);
        throw;
#endif
    }

#ifdef NDEBUG
    return default_result;
#endif
}

inline void exception_handler(cldnn_status default_error, cldnn_status* status, std::function<void()> func)
{
    //NOTE for implementer: status should not be modified after successful func() call
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        func();
    }
    catch (const cldnn::error& err)
    {
        if (status)
          *status = err.status();
        cldnn::last_err::instance().set_last_exception(err);
#ifndef NDEBUG
        throw;
#endif
    }
    catch (const std::exception& err)
    {
        if (status)
            *status = default_error;
        cldnn::last_err::instance().set_last_exception(err);

#ifndef NDEBUG
        throw;
#endif
    }
    catch (...)
    {
        if (status)
            *status = default_error;
        cldnn::last_err::instance().set_last_error_message("error unknown");
#ifndef NDEBUG
        throw;
#endif
    }
}
