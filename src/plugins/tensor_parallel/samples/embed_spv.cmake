# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Reads ${SPV_FILE} (a SPIR-V binary) and writes ${HDR_FILE} as a C header
# defining `kAllReduceSumSpv` (uint32_t array) and `kAllReduceSumSpvSize`.

file(READ "${SPV_FILE}" SPV_HEX HEX)
string(LENGTH "${SPV_HEX}" HEX_LEN)
math(EXPR BYTE_LEN "${HEX_LEN} / 2")

# Split hex into 4-byte little-endian words and format as 0x........, .
set(WORDS "")
set(I 0)
while(I LESS HEX_LEN)
    string(SUBSTRING "${SPV_HEX}" ${I} 8 W)
    # ocloc emits raw bytes; reverse byte order to form a little-endian uint32_t.
    string(SUBSTRING "${W}" 0 2 B0)
    string(SUBSTRING "${W}" 2 2 B1)
    string(SUBSTRING "${W}" 4 2 B2)
    string(SUBSTRING "${W}" 6 2 B3)
    set(WORD "0x${B3}${B2}${B1}${B0}")
    list(APPEND WORDS "${WORD}")
    math(EXPR I "${I} + 8")
endwhile()

# Pretty-print 8 words per line.
set(BODY "")
set(IDX 0)
foreach(W ${WORDS})
    if(BODY STREQUAL "")
        set(BODY "    ${W}")
    elseif(IDX EQUAL 0)
        string(APPEND BODY ",\n    ${W}")
    else()
        string(APPEND BODY ", ${W}")
    endif()
    math(EXPR IDX "(${IDX} + 1) % 8")
endforeach()

list(LENGTH WORDS WORD_COUNT)

file(WRITE "${HDR_FILE}"
"// Auto-generated from allreduce_sum.cl by ocloc; do not edit.\n"
"#pragma once\n"
"#include <cstdint>\n"
"#include <cstddef>\n"
"static const uint32_t kAllReduceSumSpv[] = {\n"
"${BODY}\n"
"};\n"
"static const size_t kAllReduceSumSpvSize = ${WORD_COUNT} * sizeof(uint32_t);\n")
