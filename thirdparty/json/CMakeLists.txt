# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# suppress shadowing names warning
set(JSON_SystemInclude ON)
set(JSON_Install ON)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)
set_target_properties(nlohmann_json PROPERTIES FOLDER thirdparty)

set(BUILD_SHARED_LIBS OFF)
add_subdirectory(nlohmann_json_schema_validator EXCLUDE_FROM_ALL)
set_target_properties(nlohmann_json_schema_validator PROPERTIES FOLDER thirdparty)

# WA for nlohmann_json_schema_validator:
# nlohmann_json must be in the same export set of nlohmann_json_schema_validator targets
install(TARGETS nlohmann_json EXPORT nlohmann_json_schema_validatorTargets
        ARCHIVE DESTINATION lib EXCLUDE_FROM_ALL)
