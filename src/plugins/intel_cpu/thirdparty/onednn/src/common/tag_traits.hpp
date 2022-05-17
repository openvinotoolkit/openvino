/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_TAG_TRAITS_HPP
#define COMMON_TAG_TRAITS_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

enum class block_dim_t {
    _,
    _A,
    _B,
    _C,
    _D,
    _E,
    _AB,
    _BC,
    _CD,
    _CE,
};

enum class inner_blk_t {
    _,
    _4a,
    _4b,
    _8a,
    _8b,
    _16a,
    _16b,
    _32a,
    _32b,
    _32c,
    _32d,
    _32e,

    _4a4b,
    _4b4a,
    _4b4c,
    _4c4b,
    _8a8b,
    _8b8a,
    _8a32b,
    _8b8c,
    _8c8b,
    _16a16b,
    _16a32b,
    _16a48b,
    _16a64b,
    _16b64a,
    _16b32a,
    _16b16a,
    _16b16c,
    _16c16b,
    _32a32b,
    _16a2b,
    _16a4b,
    _16b2c,
    _16b4c,
    _32d4c,
    _32e2c,
    _32e4c,
    _32c2e,

    _2c8b4c,
    _8a16b2a,
    _4b16a4b,
    _4b32a4b,
    _4b64a4b,
    _2b8a4b,
    _8b16a2b,
    _8b32a2b,
    _8b64a2b,
    _8b16c2b,
    _4c16b4c,
    _8c16b2c,
    _2b4c2b,
    _2c4b2c,
    _4b8c2b,
    _4c8b2c,

    _16a16b2a,
    _16a32b2a,
    _16a48b2a,
    _16a64b2a,
    _16b16a2b,
    _16b32a2b,
    _16b48a2b,
    _16b64a2b,
    _16a16b4a,
    _16a32b4a,
    _16a48b4a,
    _16a64b4a,
    _16b16a4b,
    _16b32a4b,
    _16b48a4b,
    _16b64a4b,
    _16b16c2b,
    _16c16b2c,
    _16c16b4c,
    _2a8b8a2b,
    _2b8c8b2c,
    _4a8b8a4b,
    _4b8c8b4c,
    _16c32b2c,
    _16c48b2c,
    _16c64b2c,
    _16c32b4c,
    _16c48b4c,
    _16c64b4c,
};

/** returns the offset within the block for weights blocked over oc and ic */
template <inner_blk_t f>
constexpr int AB_or_BC_blk_off(int x0, int x1) {
    using ib = inner_blk_t;
    static_assert(
            utils::one_of(f, ib::_4a4b, ib::_4b4a, ib::_4b4c, ib::_4c4b,
                    ib::_8a8b, ib::_8a32b, ib::_16a32b, ib::_8b8a, ib::_8b8c, ib::_8c8b, ib::_16a16b,
                    ib::_16b64a, ib::_16b32a, ib::_16b16a, ib::_16b16c,
                    ib::_16c16b, ib::_32a32b, ib::_16a2b, ib::_16a4b,
                    ib::_16b2c, ib::_16b4c, ib::_2c8b4c, ib::_8a16b2a,
                    ib::_4b64a4b, ib::_4b32a4b, ib::_4b16a4b, ib::_2b8a4b,
                    ib::_8b64a2b, ib::_8b32a2b, ib::_8b16a2b, ib::_8b16c2b,
                    ib::_4c16b4c, ib::_8c16b2c, ib::_2b4c2b, ib::_2c4b2c,
                    ib::_4b8c2b, ib::_4c8b2c, ib::_16a32b, ib::_16a48b,
                    ib::_16a64b, ib::_16a16b2a, ib::_16a32b2a, ib::_16a48b2a,
                    ib::_16a64b2a, ib::_16a16b4a, ib::_16a32b4a, ib::_16a48b4a,
                    ib::_16a64b4a, ib::_16b16a2b, ib::_16b16a4b, ib::_16b16c2b,
                    ib::_16c16b2c, ib::_16c16b4c, ib::_2a8b8a2b, ib::_2b8c8b2c,
                    ib::_4a8b8a4b, ib::_4b8c8b4c, ib::_16b32a2b, ib::_16b48a2b,
                    ib::_16b64a2b, ib::_16b32a4b, ib::_16b48a4b, ib::_16b64a4b,
                    ib::_16c32b2c, ib::_16c48b2c, ib::_16c64b2c, ib::_16c32b4c,
                    ib::_16c48b4c, ib::_16c64b4c),
            "unexpected inner_blk format");

    // clang-format off
    return false ? 0
        : (f == ib::_4a4b || f == ib::_4b4c) ? 4 * x0 + x1
        : (f == ib::_4b4a || f == ib::_4c4b) ? 4 * x1 + x0
        : (f == ib::_8a8b || f == ib::_8a32b || f == ib::_8b8c) ? 8 * x0 + x1
        : (f == ib::_8b8a || f == ib::_8c8b) ? 8 * x1 + x0
        : (f == ib::_16a16b || f == ib::_16a32b || f == ib::_16b16c) ? 16 * x0 + x1
        : (f == ib::_16b64a) ? 64 * x1 + x0
        : (f == ib::_16b32a) ? 32 * x1 + x0
        : (f == ib::_16b16a || f == ib::_16c16b) ? 16 * x1 + x0
        : (f == ib::_16a2b || f == ib::_16b2c) ? 2 * x0 + x1
        : (f == ib::_16a4b || f == ib::_16b4c) ? 4 * x0 + x1
        : (f == ib::_32a32b || f == ib::_16a32b) ? 32 * x0 + x1
        : (utils::one_of(f, ib::_8a16b2a, ib::_8b16c2b, ib::_16a16b2a, ib::_16b16c2b)) ? (x0 / 2) * 32 + x1 * 2 + x0 % 2
        : (f == ib::_16a48b) ? x0 * 48 + x1
        : (f == ib::_16a64b) ? x0 * 64 + x1
        : (f == ib::_16a32b2a) ? (x0 / 2) * 64 + x1 * 2 + x0 % 2
        : (f == ib::_16a48b2a) ? (x0 / 2) * 96 + x1 * 2 + x0 % 2
        : (f == ib::_16a64b2a) ? (x0 / 2) * 128 + x1 * 2 + x0 % 2
        : (f == ib::_16a16b4a) ? (x0 / 4) * 64 + x1 * 4 + x0 % 4
        : (f == ib::_16a32b4a) ? (x0 / 4) * 128 + x1 * 4 + x0 % 4
        : (f == ib::_16a48b4a) ? (x0 / 4) * 192 + x1 * 4 + x0 % 4
        : (f == ib::_16a64b4a) ? (x0 / 4) * 256 + x1 * 4 + x0 % 4
        : (f == ib::_4b16a4b || f == ib::_4c16b4c) ? (x1 / 4) * 64 + x0 * 4 + x1 % 4
        : (f == ib::_4b32a4b) ? (x1 / 4) * 128 + x0 * 4 + x1 % 4
        : (f == ib::_4b64a4b) ? (x1 / 4) * 256 + x0 * 4 + x1 % 4
        : (f == ib::_2b8a4b || f == ib::_2c8b4c) ? (x1 / 4) * 32 + x0 * 4 + x1 % 4
        : (f == ib::_16b16a2b || f == ib::_16c16b2c) ? (x1 / 2) * 32 + x0 * 2 + x1 % 2
        : (f == ib::_16b16a4b || f == ib::_16c16b4c) ? (x1 / 4) * 64 + x0 * 4 + x1 % 4
        : (f == ib::_8b16a2b || f == ib::_8c16b2c) ? (x1 / 2) * 32 + x0 * 2 + x1 % 2
        : (f == ib::_8b32a2b) ? (x1 / 2) * 64 + x0 * 2 + x1 % 2
        : (f == ib::_8b64a2b) ? (x1 / 2) * 128 + x0 * 2 + x1 % 2
        : (f == ib::_2b4c2b || f == ib::_2c4b2c) ? (x0 / 2) * 8 + x1 * 2 + x0 % 2
        : (f == ib::_4b8c2b || f == ib::_4c8b2c) ? (x0 / 2) * 16 + x1 * 2 + x0 % 2
        : (f == ib::_2a8b8a2b || f == ib::_2b8c8b2c) ? (x0 / 8) * 128 + (x1 / 2) * 16 + (x0 % 8) * 2 + x1 % 2
        : (f == ib::_4a8b8a4b || f == ib::_4b8c8b4c) ? (x0 / 8) * 256 + (x1 / 4) * 32 + (x0 % 8) * 4 + x1 % 4
        : (f == ib::_16b32a2b || f == ib::_16c32b2c) ? (x1 / 2) * 64 + x0 * 2 + x1 % 2
        : (f == ib::_16b48a2b || f == ib::_16c48b2c) ? (x1 / 2) * 96 + x0 * 2 + x1 % 2
        : (f == ib::_16b64a2b || f == ib::_16c64b2c) ? (x1 / 2) * 128 + x0 * 2 + x1 % 2
        : (f == ib::_16b32a4b || f == ib::_16c32b4c) ? (x1 / 4) * 128 + x0 * 4 + x1 % 4
        : (f == ib::_16b48a4b || f == ib::_16c48b4c) ? (x1 / 4) * 192 + x0 * 4 + x1 % 4
        : (f == ib::_16b64a4b || f == ib::_16c64b4c) ? (x1 / 4) * 256 + x0 * 4 + x1 % 4
        : INT_MIN;
    // clang-format on
}

template <inner_blk_t b>
struct inner_blk_traits {
    using ib = inner_blk_t;
};

template <format_tag_t>
struct tag_traits {
    // block_dim_t block_dims;
    // inner_blk_t inner_blks;
    // int ndims;
};

#define DECL_TRAITS(_tag, _blk_fmt, _inner_blk, _ndims) \
    template <> \
    struct tag_traits<format_tag::_tag> { \
        static constexpr block_dim_t block_dims = block_dim_t::_blk_fmt; \
        static constexpr inner_blk_t inner_blks = inner_blk_t::_inner_blk; \
        static constexpr int ndims = _ndims; \
    }

DECL_TRAITS(a, _, _, 1);
DECL_TRAITS(ab, _, _, 2);
DECL_TRAITS(abc, _, _, 3);
DECL_TRAITS(abcd, _, _, 4);
DECL_TRAITS(abcde, _, _, 5);
DECL_TRAITS(abcdef, _, _, 6);
DECL_TRAITS(abcdefg, _, _, 7);
DECL_TRAITS(abcdefgh, _, _, 8);
DECL_TRAITS(abcdefghi, _, _, 9);
DECL_TRAITS(abcdefghij, _, _, 10);
DECL_TRAITS(abcdefghijk, _, _, 11);
DECL_TRAITS(abcdefghijkl, _, _, 12);
DECL_TRAITS(abcdefghijlk, _, _, 12);
DECL_TRAITS(abcdefghikj, _, _, 11);
DECL_TRAITS(abcdefghji, _, _, 10);
DECL_TRAITS(abcdefgih, _, _, 9);
DECL_TRAITS(abcdefhg, _, _, 8);
DECL_TRAITS(abcdegf, _, _, 7);
DECL_TRAITS(abcdfe, _, _, 6);
DECL_TRAITS(abced, _, _, 5);
DECL_TRAITS(abdc, _, _, 4);
DECL_TRAITS(abdec, _, _, 5);
DECL_TRAITS(abdfce, _, _, 6);
DECL_TRAITS(acb, _, _, 3);
DECL_TRAITS(acbde, _, _, 5);
DECL_TRAITS(acbdef, _, _, 6);
DECL_TRAITS(abdefc, _, _, 6);
DECL_TRAITS(acdb, _, _, 4);
DECL_TRAITS(acdeb, _, _, 5);
DECL_TRAITS(ba, _, _, 2);
DECL_TRAITS(bac, _, _, 3);
DECL_TRAITS(bacd, _, _, 4);
DECL_TRAITS(bacde, _, _, 5);
DECL_TRAITS(bca, _, _, 3);
DECL_TRAITS(bcda, _, _, 4);
DECL_TRAITS(bcdea, _, _, 5);
DECL_TRAITS(cba, _, _, 3);
DECL_TRAITS(cdba, _, _, 4);
DECL_TRAITS(dcab, _, _, 4);
DECL_TRAITS(cdeba, _, _, 5);
DECL_TRAITS(decab, _, _, 5);
DECL_TRAITS(defcab, _, _, 6);

DECL_TRAITS(Abc4a, _A, _4a, 3);
DECL_TRAITS(aBc4b, _B, _4b, 3);
DECL_TRAITS(ABc4b16a4b, _AB, _4b16a4b, 3);
DECL_TRAITS(ABc4b32a4b, _AB, _4b32a4b, 3);
DECL_TRAITS(ABc4b64a4b, _AB, _4b64a4b, 3);
DECL_TRAITS(ABc2b8a4b, _AB, _2b8a4b, 3);
DECL_TRAITS(ABc16a16b2a, _AB, _16a16b2a, 3);
DECL_TRAITS(ABc16b16a4b, _AB, _16b16a4b, 3);
DECL_TRAITS(ABc16b32a4b, _AB, _16b32a4b, 3);
DECL_TRAITS(ABc16b48a4b, _AB, _16b48a4b, 3);
DECL_TRAITS(ABc16b64a4b, _AB, _16b64a4b, 3);
DECL_TRAITS(ABc16b16a2b, _AB, _16b16a2b, 3);
DECL_TRAITS(ABc16b32a2b, _AB, _16b32a2b, 3);
DECL_TRAITS(ABc16b48a2b, _AB, _16b48a2b, 3);
DECL_TRAITS(ABc16b64a2b, _AB, _16b64a2b, 3);
DECL_TRAITS(ABc4b4a, _AB, _4b4a, 3);
DECL_TRAITS(Abcd4a, _A, _4a, 4);
DECL_TRAITS(aBcd4b, _B, _4b, 4);
DECL_TRAITS(ABcd4b4a, _AB, _4b4a, 4);
DECL_TRAITS(ABcd4a4b, _AB, _4a4b, 4);
DECL_TRAITS(aBCd4c16b4c, _BC, _4c16b4c, 4);
DECL_TRAITS(aBCd2c8b4c, _BC, _2c8b4c, 4);
DECL_TRAITS(aBCd16b16c2b, _BC, _16b16c2b, 4);
DECL_TRAITS(aBCd16c16b4c, _BC, _16c16b4c, 4);
DECL_TRAITS(aBCd16c16b2c, _BC, _16c16b2c, 4);
DECL_TRAITS(aBCdef16c16b4c, _BC, _16c16b4c, 6);
DECL_TRAITS(aBCdef16c16b2c, _BC, _16c16b2c, 6);
DECL_TRAITS(aBCd4c4b, _BC, _4c4b, 4);
DECL_TRAITS(aBCd4b4c, _BC, _4b4c, 4);
DECL_TRAITS(Abcde4a, _A, _4a, 5);
DECL_TRAITS(aBcde4b, _B, _4b, 5);
DECL_TRAITS(ABcde4b4a, _AB, _4b4a, 5);
DECL_TRAITS(ABcde4a4b, _AB, _4a4b, 5);
DECL_TRAITS(aBCde4c4b, _BC, _4c4b, 5);
DECL_TRAITS(aBcdef4b, _B, _4b, 6);
DECL_TRAITS(aBCdef4c4b, _BC, _4c4b, 6);
DECL_TRAITS(aBCdef4b4c, _BC, _4b4c, 6);
DECL_TRAITS(aBdc4b, _B, _4b, 4);
DECL_TRAITS(aBdec4b, _B, _4b, 5);
DECL_TRAITS(aBdefc4b, _B, _4b, 6);
DECL_TRAITS(Acb4a, _A, _4a, 3);
DECL_TRAITS(Acdb4a, _A, _4a, 4);
DECL_TRAITS(Acdeb4a, _A, _4a, 5);

DECL_TRAITS(AB16b16a, _AB, _16b16a, 2);
DECL_TRAITS(AB16b32a, _AB, _16b32a, 2);
DECL_TRAITS(AB16b64a, _AB, _16b64a, 2);
DECL_TRAITS(AB8b16a2b, _AB, _8b16a2b, 2);
DECL_TRAITS(AB8b32a2b, _AB, _8b32a2b, 2);
DECL_TRAITS(AB8b64a2b, _AB, _8b64a2b, 2);
DECL_TRAITS(AB4b16a4b, _AB, _4b16a4b, 2);
DECL_TRAITS(AB4b32a4b, _AB, _4b32a4b, 2);
DECL_TRAITS(AB4b64a4b, _AB, _4b64a4b, 2);
DECL_TRAITS(AB16b16a4b, _AB, _16b16a4b, 2);
DECL_TRAITS(AB16b32a4b, _AB, _16b32a4b, 2);
DECL_TRAITS(AB16b48a4b, _AB, _16b48a4b, 2);
DECL_TRAITS(AB16b64a4b, _AB, _16b64a4b, 2);
DECL_TRAITS(AB16b16a2b, _AB, _16b16a2b, 2);
DECL_TRAITS(AB16b32a2b, _AB, _16b32a2b, 2);
DECL_TRAITS(AB16b48a2b, _AB, _16b48a2b, 2);
DECL_TRAITS(AB16b64a2b, _AB, _16b64a2b, 2);

DECL_TRAITS(BA16a16b, _AB, _16a16b, 2);
DECL_TRAITS(BA16a32b, _AB, _16a32b, 2);
DECL_TRAITS(BA16a48b, _AB, _16a48b, 2);
DECL_TRAITS(BA16a64b, _AB, _16a64b, 2);
DECL_TRAITS(BA16a16b2a, _AB, _16a16b2a, 2);
DECL_TRAITS(BA16a32b2a, _AB, _16a32b2a, 2);
DECL_TRAITS(BA16a48b2a, _AB, _16a48b2a, 2);
DECL_TRAITS(BA16a64b2a, _AB, _16a64b2a, 2);
DECL_TRAITS(BA16a16b4a, _AB, _16a16b4a, 2);
DECL_TRAITS(BA16a32b4a, _AB, _16a32b4a, 2);
DECL_TRAITS(BA16a48b4a, _AB, _16a48b4a, 2);
DECL_TRAITS(BA16a64b4a, _AB, _16a64b4a, 2);

DECL_TRAITS(Abc16a, _A, _16a, 3);
DECL_TRAITS(ABc16a16b, _AB, _16a16b, 3);
DECL_TRAITS(ABc4a4b, _AB, _4a4b, 3);
DECL_TRAITS(aBc16b, _B, _16b, 3);
DECL_TRAITS(ABc16b16a, _AB, _16b16a, 3);
DECL_TRAITS(ABc16b32a, _AB, _16b32a, 3);
DECL_TRAITS(ABc16b64a, _AB, _16b64a, 3);
DECL_TRAITS(ABc8a16b2a, _AB, _8a16b2a, 3);
DECL_TRAITS(ABc8a8b, _AB, _8a8b, 3);
DECL_TRAITS(aBc8b, _B, _8b, 3);
DECL_TRAITS(ABc8b16a2b, _AB, _8b16a2b, 3);
DECL_TRAITS(ABc8b32a2b, _AB, _8b32a2b, 3);
DECL_TRAITS(ABc8b64a2b, _AB, _8b64a2b, 3);
DECL_TRAITS(BAc8a16b2a, _AB, _8a16b2a, 3);
DECL_TRAITS(ABc8b8a, _AB, _8b8a, 3);
DECL_TRAITS(Abcd16a, _A, _16a, 4);
DECL_TRAITS(Abcd8a, _A, _8a, 4);
DECL_TRAITS(ABcd16a16b, _AB, _16a16b, 4);
DECL_TRAITS(aBcd16b, _B, _16b, 4);
DECL_TRAITS(ABcd16b16a, _AB, _16b16a, 4);
DECL_TRAITS(ABcd16b32a, _AB, _16b32a, 4);
DECL_TRAITS(ABcd16b64a, _AB, _16b64a, 4);
DECL_TRAITS(aBCd16b16c, _BC, _16b16c, 4);
DECL_TRAITS(aBCd16c16b, _BC, _16c16b, 4);
DECL_TRAITS(ABcd4b16a4b, _AB, _4b16a4b, 4);
DECL_TRAITS(ABcd4b32a4b, _AB, _4b32a4b, 4);
DECL_TRAITS(ABcd4b64a4b, _AB, _4b64a4b, 4);
DECL_TRAITS(ABcd2b8a4b, _AB, _2b8a4b, 4);
DECL_TRAITS(ABcd16a16b2a, _AB, _16a16b2a, 4);
DECL_TRAITS(ABcd16b16a4b, _AB, _16b16a4b, 4);
DECL_TRAITS(ABcd16b32a4b, _AB, _16b32a4b, 4);
DECL_TRAITS(ABcd16b48a4b, _AB, _16b48a4b, 4);
DECL_TRAITS(ABcd16b64a4b, _AB, _16b64a4b, 4);
DECL_TRAITS(ABcd16b16a2b, _AB, _16b16a2b, 4);
DECL_TRAITS(ABcd16b32a2b, _AB, _16b32a2b, 4);
DECL_TRAITS(ABcd16b48a2b, _AB, _16b48a2b, 4);
DECL_TRAITS(ABcd16b64a2b, _AB, _16b64a2b, 4);
DECL_TRAITS(ABcde16b16a2b, _AB, _16b16a2b, 5);
DECL_TRAITS(ABcde16b32a2b, _AB, _16b32a2b, 5);
DECL_TRAITS(ABcde16b48a2b, _AB, _16b48a2b, 5);
DECL_TRAITS(ABcde16b64a2b, _AB, _16b64a2b, 5);
DECL_TRAITS(ABcd8a16b2a, _AB, _8a16b2a, 4);
DECL_TRAITS(ABcd8a8b, _AB, _8a8b, 4);
DECL_TRAITS(ABcd8a32b, _AB, _8a32b, 4);
DECL_TRAITS(ABcd16a32b, _AB, _16a32b, 4);
DECL_TRAITS(aBcd8b, _B, _8b, 4);
DECL_TRAITS(ABcd8b16a2b, _AB, _8b16a2b, 4);
DECL_TRAITS(ABcd8b32a2b, _AB, _8b32a2b, 4);
DECL_TRAITS(ABcd8b64a2b, _AB, _8b64a2b, 4);
DECL_TRAITS(aBCd8b16c2b, _BC, _8b16c2b, 4);
DECL_TRAITS(BAcd8a16b2a, _AB, _8a16b2a, 4);
DECL_TRAITS(ABcd8b8a, _AB, _8b8a, 4);
DECL_TRAITS(aBCd8b8c, _BC, _8b8c, 4);
DECL_TRAITS(aBCd8c16b2c, _BC, _8c16b2c, 4);
DECL_TRAITS(aCBd8b16c2b, _BC, _8b16c2b, 4);
DECL_TRAITS(aBCd8c8b, _BC, _8c8b, 4);
DECL_TRAITS(Abcde16a, _A, _16a, 5);
DECL_TRAITS(ABcde16a16b, _AB, _16a16b, 5);
DECL_TRAITS(BAcde8a16b2a, _AB, _8a16b2a, 5);
DECL_TRAITS(ABcde4b16a4b, _AB, _4b16a4b, 5);
DECL_TRAITS(ABcde4b32a4b, _AB, _4b32a4b, 5);
DECL_TRAITS(ABcde4b64a4b, _AB, _4b64a4b, 5);
DECL_TRAITS(ABcde16b16a4b, _AB, _16b16a4b, 5);
DECL_TRAITS(ABcde16b32a4b, _AB, _16b32a4b, 5);
DECL_TRAITS(ABcde16b48a4b, _AB, _16b48a4b, 5);
DECL_TRAITS(ABcde16b64a4b, _AB, _16b64a4b, 5);
DECL_TRAITS(ABcde2b8a4b, _AB, _2b8a4b, 5);
DECL_TRAITS(aBcde16b, _B, _16b, 5);
DECL_TRAITS(ABcde16b16a, _AB, _16b16a, 5);
DECL_TRAITS(ABcde16b32a, _AB, _16b32a, 5);
DECL_TRAITS(ABcde16b64a, _AB, _16b64a, 5);
DECL_TRAITS(aBCde16b16c, _BC, _16b16c, 5);
DECL_TRAITS(aBCde16c16b, _BC, _16c16b, 5);
DECL_TRAITS(aBCde4c16b4c, _BC, _4c16b4c, 5);
DECL_TRAITS(aBCde16b16c2b, _BC, _16b16c2b, 5);
DECL_TRAITS(aBCde16c16b4c, _BC, _16c16b4c, 5);
DECL_TRAITS(aBCde16c16b2c, _BC, _16c16b2c, 5);
DECL_TRAITS(Abcde8a, _A, _8a, 5);
DECL_TRAITS(ABcde8a8b, _AB, _8a8b, 5);
DECL_TRAITS(aBcde8b, _B, _8b, 5);
DECL_TRAITS(ABcde8b16a2b, _AB, _8b16a2b, 5);
DECL_TRAITS(ABcde8b32a2b, _AB, _8b32a2b, 5);
DECL_TRAITS(ABcde8b64a2b, _AB, _8b64a2b, 5);
DECL_TRAITS(aBCde8b16c2b, _BC, _8b16c2b, 5);
DECL_TRAITS(aCBde8b16c2b, _BC, _8b16c2b, 5);
DECL_TRAITS(ABcde8b8a, _AB, _8b8a, 5);
DECL_TRAITS(aBCde8b8c, _BC, _8b8c, 5);
DECL_TRAITS(aBCde2c8b4c, _BC, _2c8b4c, 5);
DECL_TRAITS(aBCde8c16b2c, _BC, _8c16b2c, 5);
DECL_TRAITS(aBCde4b4c, _BC, _4b4c, 5);
DECL_TRAITS(aBCde8c8b, _BC, _8c8b, 5);
DECL_TRAITS(aBcdef16b, _B, _16b, 6);
DECL_TRAITS(aBCdef16b16c, _BC, _16b16c, 6);
DECL_TRAITS(aBCdef16c16b, _BC, _16c16b, 6);
DECL_TRAITS(aBCdef4c16b4c, _BC, _4c16b4c, 6);
DECL_TRAITS(aBCdef2c8b4c, _BC, _2c8b4c, 6);
DECL_TRAITS(aBCdef8b8c, _BC, _8b8c, 6);
DECL_TRAITS(aBCdef8c16b2c, _BC, _8c16b2c, 6);
DECL_TRAITS(aCBdef8b16c2b, _BC, _8b16c2b, 6);
DECL_TRAITS(aBCdef8c8b, _BC, _8c8b, 6);
DECL_TRAITS(aBdc16b, _B, _16b, 4);
DECL_TRAITS(aBdC16b2c, _BC, _16b2c, 4);
DECL_TRAITS(aBdC16b4c, _BC, _16b4c, 4);
DECL_TRAITS(aBdc8b, _B, _8b, 4);
DECL_TRAITS(aBdec16b, _B, _16b, 5);
DECL_TRAITS(aBdeC16b2c, _BC, _16b2c, 5);
DECL_TRAITS(aBdeC16b4c, _BC, _16b4c, 5);
DECL_TRAITS(aBdec8b, _B, _8b, 5);
DECL_TRAITS(aBdefc16b, _B, _16b, 6);
DECL_TRAITS(aBdefC16b2c, _BC, _16b2c, 6);
DECL_TRAITS(aBdefC16b4c, _BC, _16b4c, 6);
DECL_TRAITS(aBdefc8b, _B, _8b, 6);
DECL_TRAITS(aBedc16b, _B, _16b, 5);
DECL_TRAITS(Acb16a, _A, _16a, 3);
DECL_TRAITS(AcB16a2b, _AB, _16a2b, 3);
DECL_TRAITS(AcB16a4b, _AB, _16a4b, 3);
DECL_TRAITS(Acb8a, _A, _8a, 3);
DECL_TRAITS(aCBd16b16c, _BC, _16b16c, 4);
DECL_TRAITS(aCBde16b16c, _BC, _16b16c, 5);
DECL_TRAITS(Acdb16a, _A, _16a, 4);
DECL_TRAITS(AcdB16a2b, _AB, _16a2b, 4);
DECL_TRAITS(AcdB16a4b, _AB, _16a4b, 4);
DECL_TRAITS(Acdb8a, _A, _8a, 4);
DECL_TRAITS(Acdeb16a, _A, _16a, 5);
DECL_TRAITS(AcdeB16a2b, _AB, _16a2b, 5);
DECL_TRAITS(AcdeB16a4b, _AB, _16a4b, 5);
DECL_TRAITS(Acdeb8a, _A, _8a, 5);
DECL_TRAITS(Adcb16a, _A, _16a, 4);
DECL_TRAITS(BAc16a16b, _AB, _16a16b, 3);
DECL_TRAITS(BAcd16a16b, _AB, _16a16b, 4);
DECL_TRAITS(BAcde16a16b, _AB, _16a16b, 5);
DECL_TRAITS(ABcd32a32b, _AB, _32a32b, 4);
DECL_TRAITS(BAcde16b16a, _AB, _16b16a, 5);
DECL_TRAITS(ABcd4a8b8a4b, _AB, _4a8b8a4b, 4);
DECL_TRAITS(ABcd2a8b8a2b, _AB, _2a8b8a2b, 4);
DECL_TRAITS(aBCde4b8c8b4c, _BC, _4b8c8b4c, 5);
DECL_TRAITS(aBCde2b8c8b2c, _BC, _2b8c8b2c, 5);
DECL_TRAITS(aBdec32b, _B, _32b, 5);
DECL_TRAITS(aCBdef16c16b, _BC, _16c16b, 6);
DECL_TRAITS(aCBdef16b16c, _BC, _16b16c, 6);
DECL_TRAITS(Abcdef4a, _A, _4a, 6);
DECL_TRAITS(Abcdef8a, _A, _8a, 6);
DECL_TRAITS(Abcdef16a, _A, _16a, 6);
DECL_TRAITS(aCBd16c16b, _BC, _16c16b, 4);
DECL_TRAITS(aCBde16c16b, _BC, _16c16b, 4);
DECL_TRAITS(Acdb32a, _A, _32a, 4);
DECL_TRAITS(BAcd16b16a, _AB, _16b16a, 4);
DECL_TRAITS(BAc16b16a, _AB, _16b16a, 3);
DECL_TRAITS(aBCd2b4c2b, _BC, _2b4c2b, 4);
DECL_TRAITS(aBCde2b4c2b, _BC, _2b4c2b, 5);
DECL_TRAITS(aBCdef2b4c2b, _BC, _2b4c2b, 6);
DECL_TRAITS(aBCd2c4b2c, _BC, _2c4b2c, 4);
DECL_TRAITS(aBCde2c4b2c, _BC, _2c4b2c, 5);
DECL_TRAITS(aBCdef2c4b2c, _BC, _2c4b2c, 6);
DECL_TRAITS(aBCd4b8c2b, _BC, _4b8c2b, 4);
DECL_TRAITS(aBCde4b8c2b, _BC, _4b8c2b, 5);
DECL_TRAITS(aBCdef4b8c2b, _BC, _4b8c2b, 6);
DECL_TRAITS(aBCd4c8b2c, _BC, _4c8b2c, 4);
DECL_TRAITS(aBCde4c8b2c, _BC, _4c8b2c, 5);
DECL_TRAITS(aBCdef4c8b2c, _BC, _4c8b2c, 6);
DECL_TRAITS(abDc32d, _D, _32d, 4);
DECL_TRAITS(abDC32d4c, _CD, _32d4c, 4);
DECL_TRAITS(abCd32c, _C, _32c, 4);
DECL_TRAITS(abdEc32e, _E, _32e, 5);
DECL_TRAITS(abdEC32e2c, _CE, _32e2c, 5);
DECL_TRAITS(abdEC32e4c, _CE, _32e4c, 5);
DECL_TRAITS(abdCe32c, _C, _32c, 5);
DECL_TRAITS(abdCE32c2e, _CE, _32c2e, 5);
DECL_TRAITS(decbA16a, _A, _16a, 5);

DECL_TRAITS(aBdC16c16b2c, _BC, _16c16b2c, 4);
DECL_TRAITS(aBdC16c16b4c, _BC, _16c16b4c, 4);
DECL_TRAITS(aBdC16c32b2c, _BC, _16c32b2c, 4);
DECL_TRAITS(aBdC16c32b4c, _BC, _16c32b4c, 4);
DECL_TRAITS(aBdC16c48b2c, _BC, _16c48b2c, 4);
DECL_TRAITS(aBdC16c48b4c, _BC, _16c48b4c, 4);
DECL_TRAITS(aBdC16c64b2c, _BC, _16c64b2c, 4);
DECL_TRAITS(aBdC16c64b4c, _BC, _16c64b4c, 4);
DECL_TRAITS(aBdeC16c16b2c, _BC, _16c16b2c, 5);
DECL_TRAITS(aBdeC16c16b4c, _BC, _16c16b4c, 5);
DECL_TRAITS(aBdeC16c32b2c, _BC, _16c32b2c, 5);
DECL_TRAITS(aBdeC16c32b4c, _BC, _16c32b4c, 5);
DECL_TRAITS(aBdeC16c48b2c, _BC, _16c48b2c, 5);
DECL_TRAITS(aBdeC16c48b4c, _BC, _16c48b4c, 5);
DECL_TRAITS(aBdeC16c64b2c, _BC, _16c64b2c, 5);
DECL_TRAITS(aBdeC16c64b4c, _BC, _16c64b4c, 5);
DECL_TRAITS(aBdefC16c16b2c, _BC, _16c16b2c, 6);
DECL_TRAITS(aBdefC16c16b4c, _BC, _16c16b4c, 6);
DECL_TRAITS(aBdefC16c32b2c, _BC, _16c32b2c, 6);
DECL_TRAITS(aBdefC16c32b4c, _BC, _16c32b4c, 6);
DECL_TRAITS(aBdefC16c48b2c, _BC, _16c48b2c, 6);
DECL_TRAITS(aBdefC16c48b4c, _BC, _16c48b4c, 6);
DECL_TRAITS(aBdefC16c64b2c, _BC, _16c64b2c, 6);
DECL_TRAITS(aBdefC16c64b4c, _BC, _16c64b4c, 6);

DECL_TRAITS(AcB16b16a2b, _AB, _16b16a2b, 3);
DECL_TRAITS(AcB16b16a4b, _AB, _16b16a4b, 3);
DECL_TRAITS(AcB16b32a2b, _AB, _16b32a2b, 3);
DECL_TRAITS(AcB16b32a4b, _AB, _16b32a4b, 3);
DECL_TRAITS(AcB16b48a2b, _AB, _16b48a2b, 3);
DECL_TRAITS(AcB16b48a4b, _AB, _16b48a4b, 3);
DECL_TRAITS(AcB16b64a2b, _AB, _16b64a2b, 3);
DECL_TRAITS(AcB16b64a4b, _AB, _16b64a4b, 3);
DECL_TRAITS(AcdB16b16a2b, _AB, _16b16a2b, 4);
DECL_TRAITS(AcdB16b16a4b, _AB, _16b16a4b, 4);
DECL_TRAITS(AcdB16b32a2b, _AB, _16b32a2b, 4);
DECL_TRAITS(AcdB16b32a4b, _AB, _16b32a4b, 4);
DECL_TRAITS(AcdB16b48a2b, _AB, _16b48a2b, 4);
DECL_TRAITS(AcdB16b48a4b, _AB, _16b48a4b, 4);
DECL_TRAITS(AcdB16b64a2b, _AB, _16b64a2b, 4);
DECL_TRAITS(AcdB16b64a4b, _AB, _16b64a4b, 4);
DECL_TRAITS(AcdeB16b16a2b, _AB, _16b16a2b, 5);
DECL_TRAITS(AcdeB16b16a4b, _AB, _16b16a4b, 5);
DECL_TRAITS(AcdeB16b32a2b, _AB, _16b32a2b, 5);
DECL_TRAITS(AcdeB16b32a4b, _AB, _16b32a4b, 5);
DECL_TRAITS(AcdeB16b48a2b, _AB, _16b48a2b, 5);
DECL_TRAITS(AcdeB16b48a4b, _AB, _16b48a4b, 5);
DECL_TRAITS(AcdeB16b64a2b, _AB, _16b64a2b, 5);
DECL_TRAITS(AcdeB16b64a4b, _AB, _16b64a4b, 5);

} // namespace impl
} // namespace dnnl

#endif
