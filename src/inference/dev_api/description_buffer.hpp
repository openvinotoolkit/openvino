// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines Description buffer to conviniently works with StatusCode and ResponseDesc
 * @file description_buffer.hpp
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "ie_common.h"

namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START

/**
 * @brief      A description buffer wrapping StatusCode and ResponseDesc
 * @ingroup    ie_dev_api_error_debug
 */
struct INFERENCE_ENGINE_1_0_DEPRECATED DescriptionBuffer : public std::basic_streambuf<char, std::char_traits<char>> {
    /**
     * @brief      Creeates a description buffer with parameters
     *
     * @param[in]  err   The error code
     * @param      desc  The response desc to write an error message to
     */
    DescriptionBuffer(StatusCode err, ResponseDesc* desc) : err(err) {
        init(desc);
    }

    /**
     * @brief      Constructs with StatusCode
     *
     * @param[in]  err   The StatusCode value
     */
    explicit DescriptionBuffer(StatusCode err) : err(err) {}

    /**
     * @brief      Constructs with ResponseDesc
     *
     * @param      desc  The ResponseDesc pointer
     */
    explicit DescriptionBuffer(ResponseDesc* desc) {
        init(desc);
    }

    /**
     * @brief      Constructs with parameters
     *
     * @param      pBuffer  The buffer to wrtie to.
     * @param[in]  len      The length of `pBuffer`
     */
    DescriptionBuffer(char* pBuffer, size_t len) {
        init(pBuffer, len);
    }

    /**
     * @brief      Constructs with parameters
     *
     * @param[in]  err      The StatusCode value
     * @param      pBuffer  The buffer to wrtie to.
     * @param[in]  len      The length of `pBuffer`
     */
    DescriptionBuffer(StatusCode err, char* pBuffer, size_t len) : err(err) {
        init(pBuffer, len);
    }

    /**
     * @brief      Writes to ResponseDesc stream
     *
     * @param[in]  obj   The object to write to stream
     * @tparam     T     An object type
     *
     * @return     A reference to itself
     */
    template <class T>
    DescriptionBuffer& operator<<(const T& obj) {
        if (!stream)
            return *this;
        (*stream.get()) << obj;

        return *this;
    }

    /**
     * @brief      Converts to StatusCode
     * @return     A StatusCode value
     */
    operator StatusCode() const {
        if (stream)
            stream->flush();
        return err;
    }

private:
    std::unique_ptr<std::ostream> stream;
    StatusCode err = GENERAL_ERROR;

    void init(ResponseDesc* desc) {
        if (desc == nullptr)
            return;
        init(desc->msg, sizeof(desc->msg) / sizeof(desc->msg[0]));
    }

    void init(char* ptr, size_t len) {
        if (nullptr != ptr && len > 0) {
            // set the "put" pointer the start of the buffer and record it's length.
            setp(ptr, ptr + len - 1);
        }
        stream.reset(new std::ostream(this));

        if (nullptr != ptr && len > 0) {
            ptr[len - 1] = 0;
            (*stream.get()) << ptr;
        }
    }
};
IE_SUPPRESS_DEPRECATED_END
}  // namespace InferenceEngine
