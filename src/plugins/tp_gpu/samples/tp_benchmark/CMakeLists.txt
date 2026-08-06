# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# tp_benchmark: prefill/decode latency comparator, TP_GPU against a single GPU.

set(TARGET_NAME tp_benchmark)

add_executable(${TARGET_NAME} tp_benchmark.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

# The GPU plugin dlopens libze_loader by soname.  This binary calls no L0
# symbol, so --as-needed would drop the dependency; forcing it keeps the loader
# in the process image, where dlopen resolves it by soname without a search.
# That also puts the loader's build directory on the RPATH, so running from the
# build tree needs no LD_LIBRARY_PATH.
if(TARGET LevelZero::LevelZero)
    target_link_options(${TARGET_NAME} PRIVATE "LINKER:--no-as-needed")
    target_link_libraries(${TARGET_NAME} PRIVATE LevelZero::LevelZero)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")
