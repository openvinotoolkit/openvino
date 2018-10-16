# Copyright (c) 2016 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module supports running in script mode:
#   cmake -DINTEL__SCRIPT_MODE=ON -DINTEL__CFG_NAME=<build type/configuration name>
#     [-DINTEL__CFG_MAP_PREFIX=<map prefix>] [-DINTEL__TARGETS=<list of targets>] [-D...] -P BuildExternalProject.cmake
#
# will build all specified targets with selected build type mapped with intel_map_configuration().


cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)


# ======================================================================================================
# ================================================ UTILS ===============================================
# ======================================================================================================

# Returns mapped configuration (build type) based on input configuration.
#
# Searches for variables named in following format: "<mapPrefix>__CFG_MAPS_TO__<cfg in upper>".
# If variable is found its value is used as output configuration; otherwise input configuration
# is returned.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param mapPrefix  Prefix of variables used to map configurations (build types).
# @param config     Configuration name (build type) to map.
function(intel_map_configuration retVarName mapPrefix config)
  string(TOUPPER "${config}" _config)
  set(_mapVarName "${mapPrefix}__CFG_MAPS_TO__${_config}")
  if(DEFINED "${_mapVarName}")
    set("${retVarName}" "${${_mapVarName}}" PARENT_SCOPE)
  else()
    set("${retVarName}" "${config}" PARENT_SCOPE)
  endif()
endfunction()

# Maps imported targets with specified configuration mapping.
#
# For each configuration / build type (<cfg>), it searches for variables named
# in following format: "<mapPrefix>__CFG_MAPS_TO__<cfg in upper>".
# If variable is found and its value is different than <cfg>, the value is used
# as destination configuration during mapping of all imported targets and their interface libs
# (source configuration is <cfg>) - applies MAP_IMPORTED_CONFIG_ properties.
#
# @param mapPrefix             Prefix of variables used to map configurations (build types).
# @param [target [target ...]] Imported targets which will be mapped according to mapping.
function(intel_map_imported_targets mapPrefix)
  foreach(_config ${CMAKE_CONFIGURATION_TYPES})
    string(REPLACE ";" "\;" _config "${_config}") #WA
    intel_map_configuration(_mappedConfig "${mapPrefix}" "${_config}")
    # If we have mapping check and apply mapping on imported targets.
    if(NOT (_mappedConfig STREQUAL _config))
      string(TOUPPER "${_config}"       _upperConfig)
      string(TOUPPER "${_mappedConfig}" _upperMappedConfig)

      # BFS of all link targets needed.
      set(_targets "${ARGN}")
      set(_oldTargetsLength 0)
      list(REMOVE_DUPLICATES _targets)
      list(LENGTH _targets _targetsLength)
      while(_targetsLength GREATER _oldTargetsLength)
        set(_oldTargets "${_targets}")
        set(_oldTargetsLength ${_targetsLength})

        foreach(_target ${_oldTargets})
          string(REPLACE ";" "\;" _target "${_target}") #WA

          if(TARGET "${_target}")
            get_property(_targetIsImported      TARGET "${_target}" PROPERTY IMPORTED)
            get_property(_targetOldInterface    TARGET "${_target}" PROPERTY INTERFACE_LINK_LIBRARIES)
            get_property(_targetOldCfgInterface TARGET "${_target}" PROPERTY "INTERFACE_LINK_LIBRARIES_${_upperMappedConfig}")
            get_property(_targetInterface       TARGET "${_target}" PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES)
            get_property(_targetCfgInterface    TARGET "${_target}" PROPERTY "IMPORTED_LINK_INTERFACE_LIBRARIES_${_upperMappedConfig}")
            if(_targetIsImported)
              list(APPEND _targets "${_targetOldInterface}" "${_targetOldCfgInterface}" "${_targetInterface}" "${_targetCfgInterface}")
            endif()
          endif()
        endforeach()

        list(REMOVE_DUPLICATES _targets)
        list(LENGTH _targets _targetsLength)
      endwhile()
      list(SORT _targets)

      foreach(_target ${_targets})
        string(REPLACE ";" "\;" _target "${_target}") #WA

        if(NOT (TARGET "${_target}"))
          message(AUTHOR_WARNING "\"${_target}\" does not name CMake target. It will be omitted during\n    imports mapping (\"${_config}\" -> \"${_mappedConfig}\").")
        else()
          get_property(_targetIsImported TARGET "${_target}" PROPERTY IMPORTED)
          if(NOT _targetIsImported)
            message(AUTHOR_WARNING "\"${_target}\" is not imported target. It will be omitted during\n    imports mapping (\"${_config}\" -> \"${_mappedConfig}\").")
          else()
            set_property(TARGET "${_target}" PROPERTY "MAP_IMPORTED_CONFIG_${_upperConfig}" "${_mappedConfig}")
          endif()
        endif()
      endforeach()
    endif()
  endforeach()
endfunction()

# Helper for intel_build_external_proj_prepare_cmdline() function.
set_property(GLOBAL PROPERTY INTEL__BuildExternalProject_CMAKE_LIST_FILE "${CMAKE_CURRENT_LIST_FILE}")

# Prepares command-line for building external project.
#
# @param retValName            Name of variable placeholder where result will be returned.
# @param semicolonReplace      Characters which replaces semicolon in targets list.
# @param [mapPrefix]           Prefix of variables used to map configurations (build types).
# @param [target [target ...]] Targets to build.
function(intel_build_external_proj_prepare_cmdline retVarName semicolonReplace)
  set(_isMapPrefixSet FALSE)
  set(_isTargetsSet   FALSE)
  foreach(_arg ${ARGN})
    string(REPLACE ";" "\;" _arg "${_arg}") #WA
    if(_isMapPrefixSet)
      if(_isTargetsSet)
        set(_targets       "${_targets}${semicolonReplace}${_arg}")
      else()
        set(_targets       "${_arg}")
        set(_isTargetsSet  TRUE)
      endif()
    else()
      set(_mapPrefix       "${_arg}")
      set(_isMapPrefixSet  TRUE)
    endif()
  endforeach()

  set(_cmdline "${CMAKE_COMMAND}" "-DINTEL__SCRIPT_MODE=ON" "-DINTEL__CFG_NAME=${CMAKE_CFG_INTDIR}")
  if(_isMapPrefixSet)
    list(APPEND _cmdline "-DINTEL__CFG_MAP_PREFIX=${_mapPrefix}")
  endif()
  if(_isTargetsSet)
    list(APPEND _cmdline "-DINTEL__TARGETS=${_targets}")
  endif()
  if(_isMapPrefixSet)
    foreach(_config ${CMAKE_CONFIGURATION_TYPES})
      string(REPLACE ";" "\;" _config "${_config}") #WA
      string(TOUPPER "${_config}" _upperConfig)
      set(_mapVarName "${_mapPrefix}__CFG_MAPS_TO__${_upperConfig}")

      if(DEFINED "${_mapVarName}")
        list(APPEND _cmdline "-D${_mapVarName}=${${_mapVarName}}")
      endif()
    endforeach()
  endif()
  get_property(_moduleFilePath GLOBAL PROPERTY INTEL__BuildExternalProject_CMAKE_LIST_FILE)
  list(APPEND _cmdline "-P" "${_moduleFilePath}")
  set("${retVarName}" "${_cmdline}" PARENT_SCOPE)
endfunction()


# Prepares list of cache variables to transfer (for ExternalProject_Add()'s CMAKE_CACHE_ARGS) from current
# CMake directory.
#
# @param retValName          Name of variable placeholder where result will be returned. Cache data
#                            will be stored in this variable. Additionally in "<retValName>_NAMES" variable
#                            (suffixed with _NAMES suffix) names of transferred variables will be returned.
# @param retValName          Regular expression filter which allows to exclude specific variables from transfer.
# @param cacheCfgOptsVarName Name of the variable which contains overwrites in extensions for transferred cache
#                            data. Variable should contain list of options - each option in form:
#                              -D<cache var>:<cache var type>=<cache var value>
function(intel_transfer_cache_vars retVarName excludeFilter cacheCfgOptsVarName)
  if(excludeFilter MATCHES "^$")
    set(_noExclusion YES)
  else()
    set(_noExclusion NO)
  endif()

  # Transfer all cached options connected to CMake and toolchains.
  get_cmake_property(_cachedVariables CACHE_VARIABLES)
  set(_transferredVariables "")
  set(_transferredCacheData "")
  foreach(_cachedVariable ${_cachedVariables})
    if(_noExclusion OR (NOT (_cachedVariable MATCHES "${excludeFilter}")))
      get_property(_cachedVariableType  CACHE "${_cachedVariable}" PROPERTY TYPE)
      get_property(_cachedVariableValue CACHE "${_cachedVariable}" PROPERTY VALUE)
      # Exclude all STATIC, UNINITIALIZED and CMake's INTERNAL variables.
      if((NOT (_cachedVariableType MATCHES "^STATIC$|^UNINITALIZED$")) AND (NOT ((_cachedVariableType MATCHES "^INTERNAL$") AND (_cachedVariable MATCHES "^CMAKE_|_CMAKE$"))))
        set(_transferredCacheDataEntry "-D${_cachedVariable}:${_cachedVariableType}=${_cachedVariableValue}")
        string(REPLACE ";" "\;" _transferredCacheDataEntry "${_transferredCacheDataEntry}") #WA

        list(APPEND _transferredVariables "${_cachedVariable}")
        list(APPEND _transferredCacheData "${_transferredCacheDataEntry}")
      endif()
    endif()
  endforeach()
  unset(_cachedVariables)
  unset(_cachedVariable)
  unset(_cachedVariableType)
  unset(_transferredCacheDataEntry)

  # Extend/overwrite options with explicit configure cache options.
  if(DEFINED "${cacheCfgOptsVarName}")
    foreach(_cacheCfgOpt ${${cacheCfgOptsVarName}})
      string(REPLACE ";" "\;" _cacheCfgOpt "${_cacheCfgOpt}") #WA
      string(REGEX MATCH "^(-D)?([^:=]+)" _cacheCfgOptName "${_cacheCfgOpt}")
      set(_cacheCfgOptName "${CMAKE_MATCH_2}")
      list(FIND _transferredVariables "${_cacheCfgOptName}" _cacheCfgOptIdx)
      if(_cacheCfgOptIdx LESS 0)
        list(APPEND _transferredVariables "${_cacheCfgOptName}")
        list(APPEND _transferredCacheData "${_cacheCfgOpt}")
      else()
        list(REMOVE_AT _transferredVariables ${_cacheCfgOptIdx})
        list(REMOVE_AT _transferredCacheData ${_cacheCfgOptIdx})
        list(INSERT _transferredVariables ${_cacheCfgOptIdx} "${_cacheCfgOptName}")
        list(INSERT _transferredCacheData ${_cacheCfgOptIdx} "${_cacheCfgOpt}")
      endif()
    endforeach()
  endif()
  unset(_cacheCfgOpt)
  unset(_cacheCfgOptName)
  unset(_cacheCfgOptIdx)

  set("${retVarName}"       "${_transferredCacheData}" PARENT_SCOPE)
  set("${retVarName}_NAMES" "${_transferredVariables}" PARENT_SCOPE)
endfunction()

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


if(INTEL__SCRIPT_MODE)
  set(__INTEL_ExternProjBinDir "${CMAKE_CURRENT_BINARY_DIR}")
  set(__INTEL_OrigCfgName      "${INTEL__CFG_NAME}")
  set(__INTEL_CfgName          "${__INTEL_OrigCfgName}")

  if(NOT DEFINED INTEL__CFG_NAME)
    message(FATAL_ERROR "No build type specified while doing build of external project
        located in \"${__INTEL_ExternProjBinDir}\"."
      )
  endif()

  if(DEFINED INTEL__CFG_MAP_PREFIX)
    intel_map_configuration(__INTEL_CfgName "${INTEL__CFG_MAP_PREFIX}" "${__INTEL_CfgName}")
  endif()

  message(STATUS "============================ Build of External Project =============================")
  message(STATUS "====================================================================================")
  message(STATUS "Project location:   ${__INTEL_ExternProjBinDir}")
  message(STATUS "")
  message(STATUS "Build type:         ${__INTEL_OrigCfgName} (for multi-configuration generators)")
  if(DEFINED INTEL__CFG_MAP_PREFIX)
    message(STATUS "Mapped build type:  ${__INTEL_CfgName} (mapped by \"${INTEL__CFG_MAP_PREFIX}\")")
  endif()
  message(STATUS "")
  if(DEFINED INTEL__TARGETS)
    message(STATUS "Targets:            ${INTEL__TARGETS}")
  else()
    message(STATUS "Targets:            (default)")
  endif()
  message(STATUS "====================================================================================")

  if(__INTEL_CfgName MATCHES "^\\.$")
    set(__INTEL_CfgArg)
  else()
    set(__INTEL_CfgArg "--config" "${__INTEL_CfgName}")
  endif()

  if(DEFINED INTEL__TARGETS)
    foreach(__INTEL_Target ${INTEL__TARGETS})
      string(REPLACE ";" "\;" __INTEL_Target "${__INTEL_Target}") #WA

      message(STATUS "Target:             ${__INTEL_Target}")
      message(STATUS "")
      execute_process(COMMAND "${CMAKE_COMMAND}" --build "${__INTEL_ExternProjBinDir}" --target "${__INTEL_Target}" ${__INTEL_CfgArg})
      message(STATUS "====================================================================================")
    endforeach()
  else()
    message(STATUS "Target:             (default)")
    message(STATUS "")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${__INTEL_ExternProjBinDir}" ${__INTEL_CfgArg})
    message(STATUS "====================================================================================")
  endif()

  message(STATUS "====================================================================================")

  unset(__INTEL_ExternProjBinDir)
  unset(__INTEL_OrigCfgName)
  unset(__INTEL_CfgName)
  unset(__INTEL_CfgArg)
  unset(__INTEL_Target)
endif()