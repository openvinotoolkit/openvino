# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(passes)

set(TARGET gna_transformer_app)

set(SOURCES
    main.cpp
    menu.cpp
    menu.hpp)

add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET}
    PRIVATE gflags gna_trans_lib_transformer gna_transformer_app_passes )

add_clang_format_target(${TARGET}_clang FOR_TARGETS ${TARGET})
