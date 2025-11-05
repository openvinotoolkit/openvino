#pragma once

namespace {

// FOR_EACH macros up to 16 arguments:
#define FOR_EACH_1(M, B, x1)                         M(B, x1)
#define FOR_EACH_2(M, B, x1, x2)                     M(B, x1) M(B, x2)
#define FOR_EACH_3(M, B, x1, x2, x3)                 M(B, x1) M(B, x2) M(B, x3)
#define FOR_EACH_4(M, B, x1, x2, x3, x4)             M(B, x1) M(B, x2) M(B, x3) M(B, x4)
#define FOR_EACH_5(M, B, x1, x2, x3, x4, x5)         M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5)
#define FOR_EACH_6(M, B, x1, x2, x3, x4, x5, x6)     M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6)
#define FOR_EACH_7(M, B, x1, x2, x3, x4, x5, x6, x7) M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7)
#define FOR_EACH_8(M, B, x1, x2, x3, x4, x5, x6, x7, x8) \
    M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8)
#define FOR_EACH_9(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9)
#define FOR_EACH_10(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
    M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10)
#define FOR_EACH_11(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) \
    M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10) M(B, x11)
#define FOR_EACH_12(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) \
    M(B, x1) M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10) M(B, x11) M(B, x12)
#define FOR_EACH_13(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) \
    M(B, x1)                                                                      \
    M(B, x2) M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10) M(B, x11) M(B, x12) M(B, x13)
#define FOR_EACH_14(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14) \
    M(B, x1)                                                                           \
    M(B, x2)                                                                           \
    M(B, x3) M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10) M(B, x11) M(B, x12) M(B, x13) M(B, x14)
#define FOR_EACH_15(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) \
    M(B, x1)                                                                                \
    M(B, x2)                                                                                \
    M(B, x3)                                                                                \
    M(B, x4) M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10) M(B, x11) M(B, x12) M(B, x13) M(B, x14) M(B, x15)
#define FOR_EACH_16(M, B, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) \
    M(B, x1)                                                                                     \
    M(B, x2)                                                                                     \
    M(B, x3)                                                                                     \
    M(B, x4)                                                                                     \
    M(B, x5) M(B, x6) M(B, x7) M(B, x8) M(B, x9) M(B, x10) M(B, x11) M(B, x12) M(B, x13) M(B, x14) M(B, x15) M(B, x16)

#define GET_MACRO(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, NAME, ...) NAME

#define EXPAND(x) x

#define FOR_EACH(M, B, ...)       \
    EXPAND(GET_MACRO(_0,          \
                     __VA_ARGS__, \
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
                     FOR_EACH_1)(M, B, __VA_ARGS__))

}  // namespace