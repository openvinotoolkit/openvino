// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <istream>
#include <limits>
#include <locale>
#include <optional>
#include <streambuf>
#include <string_view>
#include <type_traits>

namespace ov::util {

/**
 * @brief A custom stream buffer that provides read-only access to a string view.
 *
 * This class inherits from `std::streambuf` and is designed to facilitate input operations directly
 * on a `std::string_view` without copying the underlying string data. It allows for efficient reading and seeking
 * operations within the string view.
 *
 * @note This stream buffer is intended for input operations only.
 */
class StringViewStreamBuf : public std::streambuf {
public:
    explicit StringViewStreamBuf(std::string_view sv) {
        char* begin = const_cast<char*>(sv.data());
        setg(begin, begin, begin + sv.size());
    }

protected:
    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override {
        if (which != std::ios_base::in) {
            return off_type(-1);
        }

        switch (dir) {
        case std::ios_base::beg:
            setg(eback(), eback() + off, egptr());
            break;
        case std::ios_base::end:
            setg(eback(), egptr() + off, egptr());
            break;
        case std::ios_base::cur:
            setg(eback(), gptr() + off, egptr());
            break;
        default:
            return off_type(-1);
        }
        if (gptr() < eback() || gptr() > egptr())
            return off_type(-1);

        return gptr() - eback();
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(pos, std::ios_base::beg, which);
    }
};
}  // namespace ov::util
