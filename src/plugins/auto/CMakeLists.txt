# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_AUTO AND NOT ENABLE_MULTI)
    return()
endif()

set (TARGET_NAME "openvino_auto_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp)

# Build the optional IPF telemetry client used by the device utilization monitor.
add_subdirectory(thirdparty)
# WA for issue 141313
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    ov_add_compiler_flags(-Wno-stringop-overflow)
endif()

if(ENABLE_AUTO AND ENABLE_MULTI)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)

    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  PSEUDO_PLUGIN_FOR "MULTI")
elseif(ENABLE_AUTO)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)
elseif(ENABLE_MULTI)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)
endif()

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

# Keep IPF's config-conditional libraries out of the version OBJECT library.
# plugin.cpp does not use IPF symbols, and ov_add_version_defines() copies LINK_LIBRARIES.
if(TARGET ${TARGET_NAME}_version)
    get_target_property(_ov_auto_plugin_version_link_libs ${TARGET_NAME} LINK_LIBRARIES)
endif()

# Enable IPF support, statically linking ClientApiLib where its CRT already matches
# OpenVINO's own (no mismatch, no DLL dependency), and falling back to the dynamically
# delay-loaded ClientApi.dll only where it doesn't (currently: Debug, only while
# ENABLE_IPF_DEBUG_WITH_RELEASE_CRT=ON forces ClientApi to the Release CRT).
if(ENABLE_IPF_DEBUG_WITH_RELEASE_CRT)
    set(_ipf_use_dynamic_genex "$<CONFIG:Debug>")
else()
    # A CRT-matching ClientApiProxy.dll is available for every config, so static linking
    # works everywhere and dynamic delay-loading is never needed.
    set(_ipf_use_dynamic_genex "$<BOOL:FALSE>")
endif()

if(TARGET openvino::ipf_client_api_static AND TARGET openvino::ipf_client_api_dynamic)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        OV_AUTO_ENABLE_IPF
        $<${_ipf_use_dynamic_genex}:ClientApi_IMPORTS>)
    target_link_libraries(${TARGET_NAME} PRIVATE
        $<${_ipf_use_dynamic_genex}:openvino::ipf_client_api_dynamic>
        $<$<NOT:${_ipf_use_dynamic_genex}>:openvino::ipf_client_api_static>
        $<${_ipf_use_dynamic_genex}:delayimp.lib>)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)

    # Delay-load: a missing/incompatible ClientApi.dll must only disable IPF telemetry
    # (handled gracefully at IpfCreate() call sites), not prevent the plugin itself
    # from loading. Only meaningful in configs that actually link the dynamic target.
    target_link_options(${TARGET_NAME} PRIVATE $<${_ipf_use_dynamic_genex}:/DELAYLOAD:ClientApi.dll>)

    # Copy ClientApi.dll next to the plugin only in configs that link it dynamically;
    # packaged installs are handled by the install(TARGETS ClientApi ...) rule in thirdparty/.
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND "$<${_ipf_use_dynamic_genex}:${CMAKE_COMMAND};-E;copy_if_different;$<TARGET_FILE:ClientApi>;$<TARGET_FILE_DIR:${TARGET_NAME}>>"
        COMMAND_EXPAND_LISTS)

    message(STATUS "[AUTO] IPF support successfully enabled for ${TARGET_NAME} "
                   "(Release: static; Debug: ${ENABLE_IPF_DEBUG_WITH_RELEASE_CRT})")
else()
    message(STATUS "[AUTO] IPF support disabled")
endif()

# Restore the IPF-free libraries for the version OBJECT library.
if(TARGET ${TARGET_NAME}_version)
    set_property(TARGET ${TARGET_NAME}_version PROPERTY LINK_LIBRARIES
        ${_ov_auto_plugin_version_link_libs})
endif()

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
