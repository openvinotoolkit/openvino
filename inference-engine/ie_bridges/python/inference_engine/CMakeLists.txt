# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# If the pyx file is a C++ file, we should specify that here.

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (COPY_IE_LIBS)
  if (UNIX)
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    SET(CMAKE_INSTALL_RPATH "$ORIGIN")
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  endif (UNIX)
endif()

set_source_files_properties(
    ie_api_impl_defs.pxd
    ie_api_impl.hpp
    ie_api_impl.cpp
    ie_api.pyx
    ie_api.pxd

  PROPERTIES CYTHON_IS_CXX TRUE
)

cython_add_module(
    ie_api

    ie_api_impl_defs.pxd
    ie_api_impl.hpp
    ie_api_impl.cpp
    ie_api.pyx
)

target_link_libraries(ie_api PRIVATE IE::inference_engine)
set_target_properties(ie_api PROPERTIES CXX_STANDARD 11 LINKER_LANGUAGE CXX)

#if (NOT UNIX AND ${PYTHON_VERSION_STRING} MATCHES "^1.4")
#  set(python_subdir "python2.7")
#else()
#  set(python_subdir "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
#endif()
#
#
# Copy required build artifacts to structure which will be used in final package
#add_custom_command(TARGET ie_api POST_BUILD
#                   
#                   COMMAND ${CMAKE_COMMAND} -E make_directory
#                   ${CMAKE_SOURCE_DIR}/bin/${python_subdir}/openvino/inference_engine/
#
#                   COMMAND ${CMAKE_COMMAND} -E touch
#                   ${CMAKE_SOURCE_DIR}/bin/${python_subdir}/openvino/__init__.py)
#
#if (${WIN32})
#add_custom_command(TARGET ie_api POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy
#                   ${CMAKE_CURRENT_BINARY_DIR}/Release/ie_api.pyd ${CMAKE_SOURCE_DIR}/bin/${python_subdir}/openvino/inference_engine/
#
#                   COMMAND ${CMAKE_COMMAND} -E copy
#                   ${CMAKE_CURRENT_BINARY_DIR}/__init__.py ${CMAKE_SOURCE_DIR}/bin/${python_subdir}/openvino/inference_engine/)
#else()
#add_custom_command(TARGET ie_api POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy
#                   ${CMAKE_CURRENT_BINARY_DIR}/ie_api.so ${CMAKE_SOURCE_DIR}/bin/${python_subdir}/openvino/inference_engine/
#
#                   COMMAND ${CMAKE_COMMAND} -E copy
#                   ${CMAKE_CURRENT_BINARY_DIR}/__init__.py ${CMAKE_SOURCE_DIR}/bin/${python_subdir}/openvino/inference_engine/)
#endif()
