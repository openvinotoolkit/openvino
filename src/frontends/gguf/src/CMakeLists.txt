# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The GGUF frontend is an MVP port of llama.cpp's ggml-openvino frontend.  It is built
# only when ENABLE_OV_GGUF_FRONTEND is ON and is marked SKIP_INSTALL so it is never
# shipped in the OpenVINO release package (it is for development / testing only).
ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                SKIP_INSTALL
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev)

ov_build_target_faster(openvino_gguf_frontend PCH)

# The quant dequant code uses ov::parallel_for (openvino/core/parallel.hpp), which
# needs the threading backend (TBB) include directories and libraries.
ov_set_threading_interface_for(openvino_gguf_frontend)

# The op translators are ported verbatim from llama.cpp and contain narrowing
# conversions (size_t/uint64_t -> int/uint32_t/float).  Silence the corresponding
# warnings so the frontend builds with warnings-as-error (notably MSVC C4244/C4267).
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    target_compile_options(openvino_gguf_frontend PRIVATE -Wno-conversion)
elseif(MSVC)
    # MSVC and icx on Windows accept MSVC-style flags
    target_compile_options(openvino_gguf_frontend PRIVATE /wd4244 /wd4267)
endif()

# SKIP_INSTALL keeps the frontend out of every install component, so it never ships
# in the release package and is never placed in the OpenVINO frontend search
# directory (where FrontEndManager would try to load it).  The unit tests do not link
# this library: they compile the frontend sources directly (see tests/CMakeLists.txt),
# keeping the test binary self-contained.
