// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "checkpoint_utils.hpp"

#include <cstring>
#include <vector>

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

static const char escape1 = '\000';
static const char null_character = '\xff';
static const char separator = '\001';
static const char escape2 = '\xff';
static const char ffcharacter = '\000';
static const char escape1_separator[2] = {escape1, separator};
static const int max_signed64_length = 10;

// This array maps encoding length to header bits in the first two bytes.
static const char length_to_header_bits[1 + max_signed64_length][2] = {{0, 0},
                                                                       {'\x80', 0},
                                                                       {'\xc0', 0},
                                                                       {'\xe0', 0},
                                                                       {'\xf0', 0},
                                                                       {'\xf8', 0},
                                                                       {'\xfc', 0},
                                                                       {'\xfe', 0},
                                                                       {'\xff', 0},
                                                                       {'\xff', '\x80'},
                                                                       {'\xff', '\xc0'}};

inline bool byte_is_0_or_255(char c) {
    return (static_cast<unsigned char>(c + 1)) < 2;
}

static inline const unsigned char* char_ptr_to_unsigned_char_ptr(const char* p) {
    const void* void_ptr = static_cast<const void*>(p);
    return static_cast<const unsigned char*>(void_ptr);
}

static inline const char* unsigned_char_ptr_to_char_ptr(const unsigned char* p) {
    const void* void_ptr = static_cast<const void*>(p);
    return static_cast<const char*>(void_ptr);
}

// return a pointer to the first byte in the range "[start..limit)"
// whose value is 0 or 255 (escape1 or escape2)
inline const char* find_special_byte(const char* start, const char* limit) {
    // If these constants were ever changed, this routine needs to change
    const char* current = start;
    while (current < limit && !byte_is_0_or_255(*current)) {
        ++current;
    }
    return current;
}

// encode "source" and append to "dest", escaping special characters
inline static void encode_string_piece(std::string& dest, const std::string& source) {
    const char* current = source.data();
    const char* limit = current + source.size();
    const char* copy_start = current;
    while (true) {
        current = find_special_byte(current, limit);
        if (current >= limit)
            break;  // No more special characters that need escaping
        char c = *(current++);
        if (c == escape1) {
            dest.append(copy_start, current - copy_start - 1);
            dest.push_back(escape1);
            dest.push_back(null_character);
            copy_start = current;
        } else {
            FRONT_END_GENERAL_CHECK(c == escape2, "[TensorFlow Frontend] incorrect model: corrupted checkpoint");
            dest.append(copy_start, current - copy_start - 1);
            dest.push_back(escape2);
            dest.push_back(ffcharacter);
            copy_start = current;
        }
    }
    if (current > copy_start) {
        dest.append(copy_start, current - copy_start);
    }
}

// reverse bytes of 64-bit number
static void convert_to_big_endian64(char* dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = (v >> (56 - 8 * i)) & 0xff;
    }
}

// compute floored log2(n)
static int log2_floor32(uint32_t n) {
    if (n == 0)
        return -1;
    int log = 0;
    uint32_t value = n;
    for (int i = 4; i >= 0; --i) {
        int shift = (1 << i);
        uint32_t x = value >> shift;
        if (x != 0) {
            value = x;
            log += shift;
        }
    }
    return log;
}

// compute floored log2(n)
static int log2_floor64(uint64_t n) {
    const uint32_t topbits = static_cast<uint32_t>(n >> 32);
    if (topbits == 0) {
        // Top bits are zero, so scan in bottom bits
        return log2_floor32(static_cast<uint32_t>(n));
    } else {
        return 32 + log2_floor32(topbits);
    }
}

// calculate the encoding length in bytes of the signed number n
static inline int signed_encoding_length(int64_t n) {
    int log2 = log2_floor64(n < 0 ? ~n : n) + 1;
    return log2 / 7 + 1;
}

void write_signed_num_increasing(std::string& dest, int64_t val) {
    const uint64_t x = val < 0 ? ~val : val;
    if (x < 64) {  // fast path for encoding length == 1
        dest += length_to_header_bits[1][0] ^ static_cast<char>(val);
        return;
    }
    // buf = val in network byte order, sign extended to 10 bytes
    const char sign_byte = val < 0 ? '\xff' : '\0';
    char buf[max_signed64_length] = {
        sign_byte,
        sign_byte,
    };
    convert_to_big_endian64(buf + 2, val);
    const int len = signed_encoding_length(x);
    FRONT_END_GENERAL_CHECK(0 < len && len <= max_signed64_length,
                            "[TensorFlow Frontend] internal error: write_signed_num_increasing failed");
    char* const begin = buf + max_signed64_length - len;
    begin[0] ^= length_to_header_bits[len][0];
    begin[1] ^= length_to_header_bits[len][1];  // ok because len >= 2
    dest.append(begin, len);
}

void write_num_increasing(std::string& dest, uint64_t val) {
    // Values are encoded with a single byte length prefix, followed
    // by the actual value in big-endian format with leading 0 bytes
    // dropped.
    unsigned char buf[9];  // 8 bytes for value plus one byte for length
    int len = 0;
    while (val > 0) {
        ++len;
        buf[9 - len] = (val & 0xff);
        val >>= 8;
    }
    buf[9 - len - 1] = len;
    ++len;
    dest.append(unsigned_char_ptr_to_char_ptr(&buf[0]) + 9 - len, len);
}

std::string encode_tensor_name_slice(const std::string& name,
                                     const std::vector<int64_t>& starts,
                                     const std::vector<int64_t> lengths) {
    std::string buffer;
    // All the tensor slice keys will start with a 0
    write_num_increasing(buffer, 0);
    encode_string_piece(buffer, name);
    buffer.append(escape1_separator, 2);
    write_num_increasing(buffer, starts.size());

    FRONT_END_GENERAL_CHECK(
        starts.size() == lengths.size(),
        "[TensorFlow Frontend] internal error or inconsistent model: check consistency of checkpoint files");
    for (size_t d = 0; d < starts.size(); ++d) {
        write_signed_num_increasing(buffer, starts[d]);
        write_signed_num_increasing(buffer, lengths[d]);
    }
    return buffer;
}

uint32_t decode_fixed32(const char* ptr) {
    uint32_t result;
    std::memcpy(&result, ptr, sizeof(result));
    return result;
}

const char* get_varint32_ptr(const char* p, const char* limit, uint32_t& value) {
    if (p < limit) {
        uint32_t result = *(char_ptr_to_unsigned_char_ptr(p));
        if ((result & 128) == 0) {
            value = result;
            return p + 1;
        }
    }
    uint32_t result = 0;
    for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
        uint32_t byte = *(char_ptr_to_unsigned_char_ptr(p));
        ++p;
        if (byte & 128) {
            // More bytes are present
            result |= ((byte & 127) << shift);
        } else {
            result |= (byte << shift);
            value = result;
            return p;
        }
    }
    return nullptr;
}

const char* get_varint64_ptr(const char* p, const char* limit, uint64_t* value) {
    uint64_t result = 0;
    for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
        uint64_t byte = *(char_ptr_to_unsigned_char_ptr(p));
        ++p;
        if (byte & 128) {
            // More bytes are present
            result |= ((byte & 127) << shift);
        } else {
            result |= (byte << shift);
            *value = result;
            return p;
        }
    }
    return nullptr;
}

bool get_varint64(std::string& input, uint64_t* value) {
    const char* p = input.data();
    const char* limit = p + input.size();
    const char* q = get_varint64_ptr(p, limit, value);
    if (q == nullptr) {
        return false;
    } else {
        input = std::string(q, limit - q);
        return true;
    }
}

const char* decode_entry(const char* p,
                         const char* limit,
                         uint32_t& shared,
                         uint32_t& non_shared,
                         uint32_t& value_length) {
    if (limit - p < 3)
        return nullptr;
    shared = char_ptr_to_unsigned_char_ptr(p)[0];
    non_shared = char_ptr_to_unsigned_char_ptr(p)[1];
    value_length = char_ptr_to_unsigned_char_ptr(p)[2];
    if ((shared | non_shared | value_length) < 128) {
        // Fast path: all three values are encoded in one byte each
        p += 3;
    } else {
        if ((p = get_varint32_ptr(p, limit, shared)) == nullptr)
            return nullptr;
        if ((p = get_varint32_ptr(p, limit, non_shared)) == nullptr)
            return nullptr;
        if ((p = get_varint32_ptr(p, limit, value_length)) == nullptr)
            return nullptr;
    }

    if (static_cast<uint32_t>(limit - p) < (non_shared + value_length)) {
        return nullptr;
    }
    return p;
}
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
