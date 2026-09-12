# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED BUILD_DIR OR NOT IS_DIRECTORY "${BUILD_DIR}")
	message(FATAL_ERROR "BUILD_DIR must point to an existing directory")
endif()

if(NOT DEFINED OLD_FLAG OR OLD_FLAG STREQUAL "")
	message(FATAL_ERROR "OLD_FLAG must be specified")
endif()

if(NOT DEFINED NEW_FLAG OR NEW_FLAG STREQUAL "")
	message(FATAL_ERROR "NEW_FLAG must be specified")
endif()

set(candidate_files)
foreach(pattern IN ITEMS "*.vcxproj" "build.ninja" "CMakeFiles/*/flags.make" "CMakeFiles/**/*.rsp")
	file(GLOB_RECURSE matched_files LIST_DIRECTORIES false "${BUILD_DIR}/${pattern}")
	list(APPEND candidate_files ${matched_files})
endforeach()
list(REMOVE_DUPLICATES candidate_files)

set(patched_files)
foreach(candidate_file IN LISTS candidate_files)
	file(READ "${candidate_file}" candidate_content)
	string(FIND "${candidate_content}" "${OLD_FLAG}" old_flag_pos)
	if(old_flag_pos EQUAL -1)
		continue()
	endif()

	string(REPLACE "${OLD_FLAG}" "${NEW_FLAG}" patched_content "${candidate_content}")
	if(NOT patched_content STREQUAL candidate_content)
		file(WRITE "${candidate_file}" "${patched_content}")
		list(APPEND patched_files "${candidate_file}")
	endif()
endforeach()

if(patched_files)
	list(LENGTH patched_files patched_file_count)
	message(STATUS "Replaced '${OLD_FLAG}' with '${NEW_FLAG}' in ${patched_file_count} generated build file(s)")
else()
	message(WARNING "Did not find '${OLD_FLAG}' under ${BUILD_DIR}")
endif()
