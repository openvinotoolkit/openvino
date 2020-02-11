// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "ie_common.h"

namespace InferenceEngine {
struct DescriptionBuffer : public std::basic_streambuf<char, std::char_traits<char>> {
    std::unique_ptr<std::ostream> stream;
    StatusCode err = GENERAL_ERROR;

    DescriptionBuffer(StatusCode err, ResponseDesc* desc): err(err) {
        init(desc);
    }

    explicit DescriptionBuffer(StatusCode err): err(err) {}

    explicit DescriptionBuffer(ResponseDesc* desc) {
        init(desc);
    }

    DescriptionBuffer(char* pBuffer, size_t len) {
        init(pBuffer, len);
    }

    DescriptionBuffer(StatusCode err, char* pBuffer, size_t len): err(err) {
        init(pBuffer, len);
    }

    template <class T>
    DescriptionBuffer& operator<<(const T& obj) {
        if (!stream) return *this;
        (*stream.get()) << obj;

        return *this;
    }

    operator StatusCode() const {
        if (stream) stream->flush();
        return err;
    }

private:
    void init(ResponseDesc* desc) {
        if (desc == nullptr) return;
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
}  // namespace InferenceEngine
