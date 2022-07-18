# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# suppress shadowing names warning
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)

# disabled due to compiler issues
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
set(BUILD_SHARED_LIBS OFF)
set(JSON_VALIDATOR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(JSON_VALIDATOR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json_schema_validator EXCLUDE_FROM_ALL)

set_target_properties(nlohmann_json_schema_validator PROPERTIES FOLDER thirdparty)

# WA for CentOS with gcc < 4.9: override defitions to remove boost usage
set_property(TARGET nlohmann_json_schema_validator PROPERTY COMPILE_DEFINITIONS "")
set_property(TARGET nlohmann_json_schema_validator PROPERTY INTERFACE_LINK_LIBRARIES nlohmann_json)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    target_compile_definitions(nlohmann_json_schema_validator PRIVATE JSON_SCHEMA_NO_REGEX)
    target_compile_options(nlohmann_json_schema_validator PRIVATE -Wno-missing-field-initializers -Wno-shadow)
endif()

# WA for nlohmann_json_schema_validator:
# nlohmann_json must be in the same export set of nlohmann_json_schema_validator targets
install(TARGETS nlohmann_json EXPORT nlohmann_json_schema_validatorTargets
        ARCHIVE DESTINATION lib EXCLUDE_FROM_ALL)
