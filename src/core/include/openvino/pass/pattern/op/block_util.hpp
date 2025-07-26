#pragma once

#include <memory>
#include <openvino/core/node_vector.hpp>

namespace {
// Registers one anchor by name
#define MAKE_ANCHOR(x) block->register_anchor(#x, x);

// FOR_EACH macros up to 16 arguments (no semicolons here!)
#define FOR_EACH_1(M, x1)                                 M(x1)
#define FOR_EACH_2(M, x1, x2)                             M(x1) M(x2)
#define FOR_EACH_3(M, x1, x2, x3)                         M(x1) M(x2) M(x3)
#define FOR_EACH_4(M, x1, x2, x3, x4)                     M(x1) M(x2) M(x3) M(x4)
#define FOR_EACH_5(M, x1, x2, x3, x4, x5)                 M(x1) M(x2) M(x3) M(x4) M(x5)
#define FOR_EACH_6(M, x1, x2, x3, x4, x5, x6)             M(x1) M(x2) M(x3) M(x4) M(x5) M(x6)
#define FOR_EACH_7(M, x1, x2, x3, x4, x5, x6, x7)         M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7)
#define FOR_EACH_8(M, x1, x2, x3, x4, x5, x6, x7, x8)     M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8)
#define FOR_EACH_9(M, x1, x2, x3, x4, x5, x6, x7, x8, x9) M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9)
#define FOR_EACH_10(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10)
#define FOR_EACH_11(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10) M(x11)
#define FOR_EACH_12(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10) M(x11) M(x12)
#define FOR_EACH_13(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10) M(x11) M(x12) M(x13)
#define FOR_EACH_14(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10) M(x11) M(x12) M(x13) M(x14)
#define FOR_EACH_15(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10) M(x11) M(x12) M(x13) M(x14) M(x15)
#define FOR_EACH_16(M, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) \
    M(x1) M(x2) M(x3) M(x4) M(x5) M(x6) M(x7) M(x8) M(x9) M(x10) M(x11) M(x12) M(x13) M(x14) M(x15) M(x16)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, NAME, ...) NAME

#define FOR_EACH(M, ...)   \
    GET_MACRO(__VA_ARGS__, \
              FOR_EACH_16, \
              FOR_EACH_15, \
              FOR_EACH_14, \
              FOR_EACH_13, \
              FOR_EACH_12, \
              FOR_EACH_11, \
              FOR_EACH_10, \
              FOR_EACH_9,  \
              FOR_EACH_8,  \
              FOR_EACH_7,  \
              FOR_EACH_6,  \
              FOR_EACH_5,  \
              FOR_EACH_4,  \
              FOR_EACH_3,  \
              FOR_EACH_2,  \
              FOR_EACH_1)  \
    (M, __VA_ARGS__)

}  // namespace

// Public macro
#define REGISTER_ANCHORS(block, ...)       \
    do {                                   \
        FOR_EACH(MAKE_ANCHOR, __VA_ARGS__) \
    } while (0)
