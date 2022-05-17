#===============================================================================
# Copyright 2016-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

file(GLOB HEADERS_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB HEADERS_SUBDIR
    ${PROJECT_BINARY_DIR}/include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.hpp
    )
include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR})

if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    add_definitions_with_host_compiler(-DDNNL_DLL_EXPORTS)
endif()

# propagate SRC specific flags
append(CMAKE_C_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")

# propagate no warning flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_SRC_CXX_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

if(NOT DNNL_VERBOSE)
    add_definitions_with_host_compiler(-DDISABLE_VERBOSE)
endif()

if(DNNL_ENABLE_CONCURRENT_EXEC)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_CONCURRENT_EXEC)
endif()

if(DNNL_ENABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is enabled")
else()
    add_definitions_with_host_compiler(-DDNNL_DISABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is disabled")
endif()

if(DNNL_ENABLE_JIT_PROFILING OR DNNL_ENABLE_ITT_TASKS)
    if (UNIX AND NOT APPLE)
        # Not every compiler adds -ldl automatically
        list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
    endif()
endif()


if(DNNL_ENABLE_ITT_TASKS AND NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    # Only supported for certain architectures (see src/common/CMakeLists.txt)
    if(DNNL_TARGET_ARCH STREQUAL "AARCH64" OR DNNL_TARGET_ARCH STREQUAL "X64")
        add_definitions_with_host_compiler(-DDNNL_ENABLE_ITT_TASKS)
    endif()
endif()

if(DNNL_ENABLE_MAX_CPU_ISA)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_MAX_CPU_ISA)
endif()

if(DNNL_ENABLE_CPU_ISA_HINTS)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_CPU_ISA_HINTS)
endif()

if(WIN32)
    add_definitions_with_host_compiler(-D_WIN)
    add_definitions_with_host_compiler(-DNOMINMAX)
endif()

# Windows does not support weak/strong symbols and no guarrantees by the linker
# for out_of_memory testing to work. Not tested on macOS
if(UNIX)
    if(DNNL_ENABLE_MEM_DEBUG)
        add_definitions_with_host_compiler(-DDNNL_ENABLE_MEM_DEBUG)
    endif()
endif()

macro(enable_conditional_compilation4 target)
    if(COMMAND ie_mark_target_as_cc)
        ie_mark_target_as_cc(${target})
        if(SELECTIVE_BUILD STREQUAL "ON")
            # After disabling a block of code, some variables might be unused.
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
                    OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
                target_compile_options(${target} PRIVATE -Wno-unused-variable)
            endif()
        endif()
    endif()
endmacro()

add_subdirectory(common)

if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(cpu)
endif()

if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(gpu)
endif()

if(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
endif()

get_property(LIB_DEPS GLOBAL PROPERTY DNNL_LIB_DEPS)
add_library(${DNNL_LIBRARY_NAME} ${DNNL_LIBRARY_TYPE}
    ${VERSION_RESOURCE_FILE} ${HEADERS_ROOT} ${HEADERS_SUBDIR} ${LIB_DEPS})
# LINK_PRIVATE for cmake 2.8.11 compatibility
target_link_libraries(${DNNL_LIBRARY_NAME} LINK_PRIVATE ${${DNNL_LIBRARY_NAME}_INTERFACE})

set_property(TARGET ${DNNL_LIBRARY_NAME} PROPERTY VERSION "${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
set_property(TARGET ${DNNL_LIBRARY_NAME} PROPERTY SOVERSION "${DNNL_VERSION_MAJOR}")

target_include_directories(${DNNL_LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    # $<INSTALL_PREFIX> is required for compatibility with cmake 2.8
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries_build(${DNNL_LIBRARY_NAME}
    "${EXTRA_SHARED_LIBS};${EXTRA_STATIC_LIBS}")
target_link_libraries_install(${DNNL_LIBRARY_NAME} "${EXTRA_SHARED_LIBS}")
if(DNNL_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${DNNL_LIBRARY_NAME} "${EXTRA_STATIC_LIBS}")
endif()

set(LIB_EXPORT_NAME "${DNNL_LIBRARY_NAME}-targets")
install(TARGETS ${DNNL_LIBRARY_NAME}
    EXPORT "${LIB_EXPORT_NAME}"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# If only Cmake could preserve the directory hierarchy...
foreach(header ${HEADERS_ROOT})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()
foreach(header ${HEADERS_SUBDIR})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/oneapi/dnnl/")
endforeach()

# Write version and package config files
set(LIB_CONFIG_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${DNNL_LIBRARY_NAME}")
set(LIB_VERSION_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${DNNL_LIBRARY_NAME}-config-version.cmake")
set(LIB_CONFIG_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${DNNL_LIBRARY_NAME}-config.cmake")
write_basic_package_version_file(
    "${LIB_VERSION_FILE}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "../cmake/config.cmake.in"
    "${LIB_CONFIG_FILE}"
    INSTALL_DESTINATION ${LIB_CONFIG_INSTALL_DIR})
install(FILES ${LIB_CONFIG_FILE} ${LIB_VERSION_FILE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})
string(TOUPPER "${DNNL_LIBRARY_NAME}::" LIB_NAMESPACE)
install(EXPORT ${LIB_EXPORT_NAME}
    NAMESPACE ${LIB_NAMESPACE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})


# Do not create compat symlinks to libraries and CMake config file when a custom
# library name is specified
if(DNNL_LIBRARY_NAME MATCHES "(dnnl|dnnld)")
    # Intel MKL-DNN compat cmake files
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND}
        -DDIR=\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake
        -P ${PROJECT_SOURCE_DIR}/cmake/gen_mkldnn_compat_cmakes.cmake)")

    # Intel MKL-DNN compat libraries
    if (WIN32)
        if (NOT MINGW)
            set(prefix "")
            set(ext ".lib")
        else()
            set(prefix "lib")
            if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
                set(ext ".dll.a")
            else()
                set(ext ".a")
            endif()
        endif()
        add_custom_target(compat_libs ALL
            ${CMAKE_COMMAND} -E copy
            $<TARGET_LINKER_FILE_DIR:${DNNL_LIBRARY_NAME}>/$<TARGET_LINKER_FILE_NAME:${DNNL_LIBRARY_NAME}>
            $<TARGET_LINKER_FILE_DIR:${DNNL_LIBRARY_NAME}>/${prefix}mkldnn${ext}
            # Workaround for MSB8065 warning.
            COMMAND ${CMAKE_COMMAND} -E touch "CMakeFiles/compat_libs"
            DEPENDS ${DNNL_LIBRARY_NAME})
        install(FILES $<TARGET_LINKER_FILE_DIR:${DNNL_LIBRARY_NAME}>/${prefix}mkldnn${ext}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    else()
        if(DNNL_LIBRARY_TYPE STREQUAL "SHARED")
            set_ternary(ext APPLE ".dylib" ".so")
            set(vers ".${DNNL_VERSION_MAJOR};.${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
        else()
            set(ext ".a")
            set(vers "")
        endif()
        foreach(ver "" ${vers})
            set_ternary(ext_and_ver APPLE "${ver}${ext}" "${ext}${ver}")
            get_property(lib_location TARGET ${DNNL_LIBRARY_NAME} PROPERTY LIBRARY_OUTPUT_DIRECTORY)
            if(lib_location)
                set(compat_link "${lib_location}/libmkldnn${ext_and_ver}")
            else()
                set(compat_link "${CMAKE_CURRENT_BINARY_DIR}/libmkldnn${ext_and_ver}")
            endif()
            if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)               
                add_custom_command(OUTPUT ${compat_link}
                    COMMAND ${CMAKE_COMMAND} -E copy libdnnl${ext_and_ver} ${compat_link}
                    DEPENDS ${DNNL_LIBRARY_NAME})
            else()
                add_custom_command(OUTPUT ${compat_link}
                    # to make the next command work fine
                    COMMAND ${CMAKE_COMMAND} -E remove -f ${compat_link}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink libdnnl${ext_and_ver} ${compat_link}
                    DEPENDS ${DNNL_LIBRARY_NAME})
            endif()
            add_custom_target(compat_libs${ver} ALL DEPENDS ${compat_link})
            install(FILES ${compat_link} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
        endforeach()
    endif()
endif()

# Install custom find modules for transitive dependencies
set(LIB_CONFIG_INSTALL_DIR_COMPAT "${CMAKE_INSTALL_LIBDIR}/cmake/mkldnn")
if("${DNNL_CPU_THREADING_RUNTIME}" MATCHES "^(TBB|TBB_AUTO)$")
    if(WIN32)
        install(FILES "../cmake/win/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(APPLE)
        install(FILES "../cmake/mac/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(UNIX)
        install(FILES "../cmake/lnx/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    endif()
    install(FILES "${CMAKE_INSTALL_PREFIX}/${LIB_CONFIG_INSTALL_DIR}/FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR_COMPAT})
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    install(FILES
        "../cmake/FindOpenCL.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    install(FILES "../cmake/FindOpenCL.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR_COMPAT})
endif()

if(DNNL_WITH_SYCL)
    install(FILES
        "../cmake/FindLevelZero.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

# On Windows we need to add dnnl.dll path to CTESTCONFIG_PATH which is later
# passed to ctest and Visual Studio solutions
if(WIN32)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
            append_to_windows_path_list(CTESTCONFIG_PATH
                "${CMAKE_CURRENT_BINARY_DIR}/${BUILD_TYPE}")
        endforeach()
    else()
        append_to_windows_path_list(CTESTCONFIG_PATH
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
endif()
