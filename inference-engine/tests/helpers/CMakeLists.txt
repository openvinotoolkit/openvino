# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME helpers)

file(GLOB HELPERS_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file (GLOB HELPERS_INCLUDES
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
      )

## Enable Models multiple search pathes
message("configuring file: ${CMAKE_CURRENT_BINARY_DIR}/test_model_repo.h")
configure_file(test_model_repo.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/test_model_repo.hpp @ONLY)

function(add_helpers target_name)
    add_library(${target_name} STATIC ${HELPERS_SRC})

    # detecting regex support
    if(USE_BOOST_RE)
        target_compile_definitions(${target_name} PUBLIC USE_BOOST_RE)

        debug_message(STATUS "Adding boost dependency")
        if (VERBOSE_BUILD)
            set(Boost_DEBUG on)
        endif ()
        find_package(Boost REQUIRED COMPONENTS regex)
        target_link_libraries(${target_name} PUBLIC ${Boost_REGEX_LIBRARY})
        target_include_directories(${target_name} PUBLIC ${Boost_INCLUDE_DIRS})
    endif()

    if(MSVC)
        set(PUGI pugixml_mt)
    else()
        set(PUGI pugixml)
    endif()

    target_link_libraries(${target_name} PRIVATE ${PUGI} PUBLIC gtest)

    target_include_directories(${target_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                                     $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
                                                     "${IE_MAIN_SOURCE_DIR}/src/vpu/"
                                                     "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src"
                                             PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: eliminate dependency on samples
    target_include_directories(${target_name} PUBLIC
        "${IE_MAIN_SOURCE_DIR}/samples/common/os/windows")

    target_compile_definitions(${target_name} PUBLIC ${ARGV}
        MODELS_PATH=\"${MODELS_PATH}\" DATA_PATH=\"${VALIDATION_SET}\")

    set_property(TARGET ${target_name} PROPERTY COMPILE_PDB_NAME ${target_name})

    # add_cpplint_target(${target_name}_cpplint FOR_TARGETS ${target_name})
endfunction()

add_helpers(${TARGET_NAME})
add_helpers(${TARGET_NAME}_s USE_STATIC_IE)
