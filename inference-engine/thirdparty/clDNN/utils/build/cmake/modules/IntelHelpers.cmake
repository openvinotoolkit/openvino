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



#cmake_policy(PUSH) # Not needed... CMake manages policy scope.
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)


# TODO: Check get_property() with CMake cache interference. According to doc it should return empty value, but it may unset variable in some conditions.

# ======================================================================================================
# ================================================ UTILS ===============================================
# ======================================================================================================

# ================================================ Flags ===============================================

# Escapes text for regular expressions.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param text       Text to escape.
function(intel_regex_escape retVarName text)
  string(REGEX REPLACE "\\^|\\$|\\.|\\-|\\[|\\]|\\(|\\)|\\+|\\*|\\?|\\||\\\\" "\\\\\\0" _escapedText "${text}")
  set("${retVarName}" "${_escapedText}" PARENT_SCOPE)
endfunction()


# Removes flags from variable which match specified regular expression.
#
# @param flagsVarName      Name of variable with flags. The variable will be updated.
# @param [flag [flag ...]] Regular expressions which describe flags to remove.
function(intel_flag_remove_re flagsVarName)
  set(_updatedFlags "${${flagsVarName}}")
  foreach(_flag ${ARGN})
    string(REPLACE ";" "\;" _flag "${_flag}") # [WA#1] Must escape ; again if occurred in flag.
    string(REGEX REPLACE "([ ]+|^)(${_flag})([ ]+|$)" " " _updatedFlags "${_updatedFlags}")
  endforeach()
  string(STRIP "${_updatedFlags}" _updatedFlags)

  if(_updatedFlags MATCHES "^$") # [WA#3] Empty string sometimes unsets variable which can lead to reading of backing cached variable with the same name.
    set("${flagsVarName}" " " PARENT_SCOPE)
  else()
    set("${flagsVarName}" "${_updatedFlags}" PARENT_SCOPE)
  endif()
endfunction()


# Adds flags to variable. If flag is in variable it is omitted.
#
# @param flagsVarName      Name of variable with flags. The variable will be updated.
# @param [flag [flag ...]] Flags which will be added to variable (if they are not in it already).
function(intel_flag_add_once flagsVarName)
  set(_updatedFlags "${${flagsVarName}}")
  foreach(_flag ${ARGN})
    string(REPLACE ";" "\;" _flag "${_flag}") # [WA#1] Must escape ; again if occurred in flag.
    intel_regex_escape(_escapedFlag "${_flag}")
    if(NOT (_updatedFlags MATCHES "([ ]+|^)(${_escapedFlag})([ ]+|$)"))
      set(_updatedFlags "${_updatedFlags} ${_flag}")
    endif()
  endforeach()
  string(STRIP "${_updatedFlags}" _updatedFlags)

  set("${flagsVarName}" "${_updatedFlags}" PARENT_SCOPE)
endfunction()


# Removes flags from variable which match specified flag.
#
# @param flagsVarName      Name of variable with flags. The variable will be updated.
# @param [flag [flag ...]] Strings which are equal to flags to remove.
function(intel_flag_remove flagsVarName)
  set(_updatedFlags "${${flagsVarName}}")
  foreach(_flag ${ARGN})
    string(REPLACE ";" "\;" _flag "${_flag}") # [WA#1] Must escape ; again if occurred in flag.
    intel_regex_escape(_escapedFlag "${_flag}")
    intel_flag_remove_re(_updatedFlags "${_escapedFlag}")
  endforeach()

  set("${flagsVarName}" "${_updatedFlags}" PARENT_SCOPE)
endfunction()

# Adds flags to flag property of target (if they are not added already).
#
# @param targetName        Name of target.
# @param propertyName      String with property name.
# @param [flag [flag ...]] Flags which will be added to property (if they are not in it already).
function(intel_target_flag_property_add_once targetName propertyName)
  get_property(_flagsExist TARGET "${targetName}" PROPERTY "${propertyName}" DEFINED)
  if(NOT _flagsExist)
    message(AUTHOR_WARNING "Property \"${propertyName}\" is not defined.")
  endif()

  get_property(_flags TARGET "${targetName}" PROPERTY "${propertyName}")
  intel_flag_add_once(_flags "${ARGN}") # [WA#2] To handle ; correctly some lists must be put in string form.
  set_property(TARGET "${targetName}" PROPERTY "${propertyName}" "${_flags}")
endfunction()


# Adds flags to flag property of target (if they are not added already).
#
# Flags are read from variable (if defined). The base property is updated by base variable and
# all per-configuration properties (with _<CONFIG> suffix) are updated by variables named from
# base name with _<CONFIG> suffix appended.
#
# @param targetName        Name of target.
# @param propertyBaseName  String with property name where add flags to (base name).
# @param varBaseName       Variable name where list of flags is stored (base name).
function(intel_target_flag_property_add_once_config_var targetName propertyBaseName varBaseName)
  if(DEFINED "${varBaseName}")
    intel_target_flag_property_add_once("${targetName}" "${propertyBaseName}" "${${varBaseName}}")  # [WA#2] To handle ; correctly some lists must be put in string form.
  endif()

  foreach(_configName ${CMAKE_CONFIGURATION_TYPES})
    string(REPLACE ";" "\;" _configName "${_configName}") # [WA#1] Must escape ; again if occurred in item.
    string(TOUPPER "${_configName}" _upperConfigName)
    set(_propertyName  "${propertyBaseName}_${_upperConfigName}")
    set(_varName       "${varBaseName}_${_upperConfigName}")

    if(DEFINED "${_varName}")
      intel_target_flag_property_add_once("${targetName}" "${_propertyName}" "${${_varName}}") # [WA#2] To handle ; correctly some lists must be put in string form.
    endif()

    unset(_upperConfigName)
    unset(_propertyName)
    unset(_varName)
  endforeach()
endfunction()

# Registers setting and relations on flag options which are usually connected to compiler / linker options.
#
# Settings on the same domain can be set on different calls of this function. Groups with the same
# name will be overwritten (by last GROUP entry of last call of function).
# The same overwrite behavior is defined for aliases with the same name.
#
# intel_flag_register_settings(
#     settingsDomainName
#     [GROUP [NO_REGEX] [NAME groupName] groupedFlag [groupedFlag [...]]]
#     [GROUP [NO_REGEX] [NAME groupName] groupedFlag [groupedFlag [...]]]
#     [...]
#     [ALIAS aliasName [aliasedFlag [{ALLOW_MULTIPLE | REMOVE_GROUP }]]]
#     [ALIAS aliasName [aliasedFlag [{ALLOW_MULTIPLE | REMOVE_GROUP }]]]
#     [...]
#   )
#
# GROUP groups mutually exclusive flags. Only one flag from the group can be applied. NO_REGEX indicates
# that grouped flags are described directly (not as regular expression patterns). NAME allows to create
# named group.
# 
# ALIAS allows to apply flags using unified name. ALLOW_MULTIPLE indicates that flag can be applied multiple times.
# ALLOW_MULTIPLE only works for non-grouped aliased flags. REMOVE_GROUP treats aliasedFlag as group name which
# flags will be removed when alias will be applied.
#
# @param settingsDomainName Domain name for settings / relations. The domain name allows to differentate and
#                           to create multiple sets of relations. All operations on settings functions use
#                           domain name to identify relations domain.
# @param groupName          Optional name of the group. Named groups can be overwritten.
# @param groupedFlag        Flags which are defined in specified group. Flags in the group are treated as
#                           mutually exclusive.
# @param aliasName          Name of alias for the flag.
# @param aliasedFlag        Raw value of flag. It contains group name when REMOVE_GROUP is specified.
function(intel_flag_register_settings settingsDomainName)
  set(_settingsPropPrefix    "INTEL_CMAKE_PROPERTY__FLAG_SETTINGS_D${settingsDomainName}_")
  set(_namedGroupsPropName   "${_settingsPropPrefix}PNAMED_GROUPS")
  set(_groupIdxCountPropName "${_settingsPropPrefix}PGROUP_INDEX")
  set(_aliasesPropName       "${_settingsPropPrefix}PALIASES")

  # Preserve group index to support multiple registration calls.
  get_property(_groupIdxSet GLOBAL PROPERTY "${_groupIdxCountPropName}" SET)
  if(_groupIdxSet)
    get_property(_groupIdx GLOBAL PROPERTY "${_groupIdxCountPropName}")
  else()
    set(_groupIdx 0)
    set_property(GLOBAL PROPERTY "${_groupIdxCountPropName}" "${_groupIdx}")
  endif()
  # Use named groups to verify connections.
  get_property(_namedGroups GLOBAL PROPERTY "${_namedGroupsPropName}")

  set(_parseState 1)
  foreach(_settingsArg ${ARGN})
    string(REPLACE ";" "\;" _settingsArg "${_settingsArg}")  # [WA#1] Must escape ; again if occurred in item.

    # States: [0] <param> [1] *( "GROUP" [2] *1( "NO_REGEX" [3] ) *1( "NAME" [4] <param> [5] ) <param> [6] *<param> [6] )
    #         *( "ALIAS" [7] <param> [8] *1( <param> [9] *<param> [9] *1( { "ALLOW_MULTIPLE" | "REMOVE_GROUP" } [10] ) ) )
    # Transitions: 0 -> 1 // by explict parameter
    #              1 (GROUP) -> 2
    #              1 (ALIAS) -> 7
    #              2 (NO_REGEX) -> 3
    #              2 (NAME) -> 4
    #              2 -> 6
    #              3 (NAME) -> 4
    #              3 -> 6
    #              4 -> 5 -> 6
    #              6 (GROUP) -> 2
    #              6 (ALIAS) -> 7
    #              6 -> 6
    #              7 -> 8
    #              8 (ALIAS) -> 7
    #              8 -> 9
    #              9 (ALIAS) -> 7
    #              9 (ALLOW_MULTIPLE) -> 10 (ALIAS) -> 7
    #              9 (REMOVE_GROUP)   -> 10 (ALIAS) -> 7
    #              9 -> 9
    # Stop States: 1, 6, 8, 9, 10
    if(_parseState EQUAL 1)
      if(_settingsArg MATCHES "^GROUP$")
        set(_parseState 2)
      elseif(_settingsArg MATCHES "^ALIAS$")
        set(_parseState 7)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_settingsArg}\".")
      endif()
    elseif(_parseState EQUAL 2)
      set(_groupUseRe YES)
      set(_namedGroup NO)
      set(_groupName "I${_groupIdx}")
      set(_groupedFlagsPropName "${_settingsPropPrefix}G${_groupName}_FLAGS")

      math(EXPR _groupIdx "${_groupIdx} + 1")

      if(_settingsArg MATCHES "^NO_REGEX$")
        set(_groupUseRe NO)
        set(_parseState 3)
      elseif(_settingsArg MATCHES "^NAME$")
        set(_parseState 4)
      else()
        if(NOT _groupUseRe)
          intel_regex_escape(_settingsArg "${_settingsArg}")
        endif()
        set_property(GLOBAL PROPERTY "${_groupedFlagsPropName}" "${_settingsArg}")
        set(_parseState 6)
      endif()
    elseif(_parseState EQUAL 3)
      if(_settingsArg MATCHES "^NAME$")
        set(_parseState 4)
      else()
        if(NOT _groupUseRe)
          intel_regex_escape(_settingsArg "${_settingsArg}")
        endif()
        set_property(GLOBAL PROPERTY "${_groupedFlagsPropName}" "${_settingsArg}")
        set(_parseState 6)
      endif()
    elseif(_parseState EQUAL 4)
      set(_namedGroup YES)
      set(_groupName "N${_settingsArg}")
      set(_groupedFlagsPropName "${_settingsPropPrefix}G${_groupName}_FLAGS")

      math(EXPR _groupIdx "${_groupIdx} - 1") # Named group does not have index identifier (we can reuse pre-allocated index).

      set(_parseState 5)
    elseif(_parseState EQUAL 5)
      if(NOT _groupUseRe)
        intel_regex_escape(_settingsArg "${_settingsArg}")
      endif()
      set_property(GLOBAL PROPERTY "${_groupedFlagsPropName}" "${_settingsArg}")
      set(_parseState 6)
    elseif(_parseState EQUAL 6)
      # Updating list of named groups or next available index (for unnamed groups).
      # This action should be triggered at transition to state 6 which is Stop state, so the action must be also when there is no more parameters.
      if(_namedGroup)
        list(FIND _namedGroups "${_groupName}" _namedGroupIdx)
        if(_namedGroupIdx LESS 0)
          set_property(GLOBAL APPEND PROPERTY "${_namedGroupsPropName}" "${_groupName}")
          list(APPEND _namedGroups "${_groupName}")
        endif()
      else()
        set_property(GLOBAL PROPERTY "${_groupIdxCountPropName}" "${_groupIdx}")
      endif()

      if(_settingsArg MATCHES "^GROUP$")
        set(_parseState 2)
      elseif(_settingsArg MATCHES "^ALIAS$")
        set(_parseState 7)
      else()
        if(NOT _groupUseRe)
          intel_regex_escape(_settingsArg "${_settingsArg}")
        endif()
        set_property(GLOBAL APPEND PROPERTY "${_groupedFlagsPropName}" "${_settingsArg}")
      endif()
    elseif(_parseState EQUAL 7)
      set(_aliasName "${_settingsArg}")
      set(_aliasRawPropName        "${_settingsPropPrefix}A${_aliasName}_RAW")
      set(_aliasRGroupPropName     "${_settingsPropPrefix}A${_aliasName}_REMOVE_GROUP")
      set(_aliasAllowMultiPropName "${_settingsPropPrefix}A${_aliasName}_ALLOW_MULTIPLE")

      get_property(_aliases GLOBAL PROPERTY "${_aliasesPropName}")
      list(FIND _aliases "${_aliasName}" _aliasIdx)
      if(_aliasIdx LESS 0)
        set_property(GLOBAL APPEND PROPERTY "${_aliasesPropName}" "${_aliasName}")
      endif()

      set_property(GLOBAL PROPERTY "${_aliasRawPropName}")
      set_property(GLOBAL PROPERTY "${_aliasRGroupPropName}")
      set_property(GLOBAL PROPERTY "${_aliasAllowMultiPropName}" NO)

      set(_parseState 8)
    elseif(_parseState EQUAL 8)
      if(_settingsArg MATCHES "^ALIAS$")
        set(_parseState 7)
      else()
        set_property(GLOBAL PROPERTY "${_aliasRawPropName}" "${_settingsArg}")
        set(_parseState 9)
      endif()
    elseif(_parseState EQUAL 9)
      if(_settingsArg MATCHES "^ALIAS$")
        set(_parseState 7)
      elseif(_settingsArg MATCHES "^ALLOW_MULTIPLE$")
        set_property(GLOBAL PROPERTY "${_aliasAllowMultiPropName}" YES)
        set(_parseState 10)
      elseif(_settingsArg MATCHES "^REMOVE_GROUP$")
        get_property(_groupsToRemove GLOBAL PROPERTY "${_aliasRawPropName}")
        set_property(GLOBAL PROPERTY "${_aliasRawPropName}")
        foreach(_groupToRemove ${_groupsToRemove})
          string(REPLACE ";" "\;" _groupToRemove "${_groupToRemove}") # [WA#1] Must escape ; again if occurred in item.
          list(FIND _namedGroups "N${_groupToRemove}" _namedGroupIdx)
          if(_namedGroupIdx LESS 0)
            message(WARNING "Named group \"${_groupToRemove}\" referenced in \"${_aliasName}\" alias does not exist yet.")
          endif()
          set_property(GLOBAL APPEND PROPERTY "${_aliasRGroupPropName}" "N${_groupToRemove}")
        endforeach()
        set(_parseState 10)
      else()
        set_property(GLOBAL APPEND PROPERTY "${_aliasRawPropName}" "${_settingsArg}")
      endif()
    elseif(_parseState EQUAL 10)
      if(_settingsArg MATCHES "^ALIAS$")
        set(_parseState 7)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_settingsArg}\".")
      endif()
    else()
      message(FATAL_ERROR "Invalid parameter token near \"${_settingsArg}\".")
    endif()
  endforeach()
  if(_parseState EQUAL 6)
    # Updating list of named groups or next available index (for unnamed groups).
    # This action should be triggered at transition to state 6 which is Stop state, so the action must be also when there is no more parameters.
    if(_namedGroup)
      list(FIND _namedGroups "${_groupName}" _namedGroupIdx)
      if(_namedGroupIdx LESS 0)
        set_property(GLOBAL APPEND PROPERTY "${_namedGroupsPropName}" "${_groupName}")
        list(APPEND _namedGroups "${_groupName}")
      endif()
    else()
      set_property(GLOBAL PROPERTY "${_groupIdxCountPropName}" "${_groupIdx}")
    endif()
  endif()
  if(NOT ((_parseState EQUAL 1) OR (_parseState EQUAL 6) OR (_parseState EQUAL 8) OR (_parseState EQUAL 9) OR (_parseState EQUAL 10)))
    message(FATAL_ERROR "Invalid number of parameters.")
  endif()
endfunction()

# Applies settings to flag variables. Settings are applied according to registered configuration
# (by intel_flag_register_settings).
#
# intel_flag_apply_settings(
#     settingsDomainName
#     flagsMainVarName
#     [FLAG         flagsVarName [flagsVarName [...]]]
#     [FLAG         flagsVarName [flagsVarName [...]]]
#     [...]
#     [SET          flag [flag [...]]]
#     [SET_RAW      flag [flag [...]]]
#     [REMOVE_GROUP groupName [groupName [...]]]
#     [{SET|SET_RAW|REMOVE_GROUP} ...]
#     [...]
#   )
#
# Allowed operation to apply:
# SET          - sets flag (takes into consideration aliases).
# SET_RAW      - sets flag (does NOT take aliases into consideration).
# REMOVE_GROUP - removes all flags identified by specified group.
# Operations are applied in definition order.
#
# @param settingsDomainName Domain name for settings / relations. The domain name allows to differentate and
#                           to create multiple sets of relations. All operations on settings functions use
#                           domain name to identify relations domain.
# @param flagsMainVarName   Name of main variable for flags. Any added flags are added to main variable.
# @param flagsVarName       Names of any additional flag variables which will be cleaned up from mutually
#                           exclusive flags (or from selected groups of flags).
# @param flag               Flags to set (SET or SET_RAW).
# @param groupName          Names of groups of flags to remove (REMOVE_GROUP).
function(intel_flag_apply_settings settingsDomainName flagsMainVarName)
  set(_settingsPropPrefix    "INTEL_CMAKE_PROPERTY__FLAG_SETTINGS_D${settingsDomainName}_")
  set(_namedGroupsPropName   "${_settingsPropPrefix}PNAMED_GROUPS")
  set(_groupIdxCountPropName "${_settingsPropPrefix}PGROUP_INDEX")
  set(_aliasesPropName       "${_settingsPropPrefix}PALIASES")

  get_property(_domainExists GLOBAL PROPERTY "${_groupIdxCountPropName}" SET)
  if(NOT _domainExists)
    message(FATAL_ERROR "Settings domain \"${settingsDomainName}\" does not exist.")
  endif()

  set(_flagVarNames  "${flagsMainVarName}")
  set(_operations    "")
  set(_flagsOrGroups "")

  set(_parseState 2)
  foreach(_settingsArg ${ARGN})
    string(REPLACE ";" "\;" _settingsArg "${_settingsArg}") # [WA#1] Must escape ; again if occurred in item.

    # States: [0] <param> [1] <param> [2] *( "FLAG" [3] <param> [4] *<param> [4] ) *( ( "SET" | "SET_RAW" | "REMOVE_GROUP" ) [5] <param> [6] *<param> [6] )
    # Transitions: 0 -> 1 -> 2 // by explict parameters
    #              2 (FLAG) -> 3
    #              2 (SET|SET_RAW|REMOVE_GROUP) -> 5
    #              3 -> 4
    #              4 (FLAG) -> 3
    #              4 (SET|SET_RAW|REMOVE_GROUP) -> 5
    #              4 -> 4
    #              5 -> 6
    #              6 (SET|SET_RAW|REMOVE_GROUP) -> 5
    #              6 -> 6
    # Stop States: 2, 4, 6
    if(_parseState EQUAL 2)
      if(_settingsArg MATCHES "^FLAG$")
        set(_parseState 3)
      elseif(_settingsArg MATCHES "^SET|SET_RAW|REMOVE_GROUP$")
        set(_opType "${CMAKE_MATCH_0}")
        set(_parseState 5)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_settingsArg}\".")
      endif()
    elseif(_parseState EQUAL 3)
      list(APPEND _flagVarNames "${_settingsArg}")
      set(_parseState 4)
    elseif(_parseState EQUAL 4)
      if(_settingsArg MATCHES "^FLAG$")
        set(_parseState 3)
      elseif(_settingsArg MATCHES "^SET|SET_RAW|REMOVE_GROUP$")
        set(_opType "${CMAKE_MATCH_0}")
        set(_parseState 5)
      else()
        list(APPEND _flagVarNames "${_settingsArg}")
      endif()
    elseif(_parseState EQUAL 5)
      list(APPEND _operations    "${_opType}")
      list(APPEND _flagsOrGroups "${_settingsArg}")
      set(_parseState 6)
    elseif(_parseState EQUAL 6)
      if(_settingsArg MATCHES "^SET|SET_RAW|REMOVE_GROUP$")
        set(_opType "${CMAKE_MATCH_0}")
        set(_parseState 5)
      else()
        list(APPEND _operations    "${_opType}")
        list(APPEND _flagsOrGroups "${_settingsArg}")
      endif()
    else()
      message(FATAL_ERROR "Invalid parameter token near \"${_settingsArg}\".")
    endif()
  endforeach()
  if(NOT ((_parseState EQUAL 2) OR (_parseState EQUAL 4) OR (_parseState EQUAL 6)))
    message(FATAL_ERROR "Invalid number of parameters.")
  endif()

  set(_updatedFlagsMainVarName "_updatedFlags_${flagsMainVarName}")
  foreach(_flagVarName ${_flagVarNames})
    string(REPLACE ";" "\;" _flagVarName "${_flagVarName}")
    set("_updatedFlags_${_flagVarName}" "${${_flagVarName}}")
  endforeach()

  get_property(_groupIdx    GLOBAL PROPERTY "${_groupIdxCountPropName}") # Next available index for unnamed group.
  get_property(_namedGroups GLOBAL PROPERTY "${_namedGroupsPropName}")
  get_property(_aliases     GLOBAL PROPERTY "${_aliasesPropName}")
  set(_groups "${_namedGroups}")
  if(_groupIdx GREATER 0)
    math(EXPR _groupIdxM1 "${_groupIdx} - 1")
    foreach(_idx RANGE 0 ${_groupIdxM1})
      list(APPEND _groups "I${_idx}")
    endforeach()
  endif()

  set(_operationIdx 0)
  foreach(_operation ${_operations}) # Operation type does not have ; -> no need for [WA#1]
    list(GET _flagsOrGroups ${_operationIdx} _flagOrGroup)
    string(REPLACE ";" "\;" _flagOrGroup "${_flagOrGroup}") # [WA#1] Must escape ; again if occurred in item.

    set(_groupsToRemove "")
    set(_flagsToAdd     "")

    set(_aliasRawPropName        "${_settingsPropPrefix}A${_flagOrGroup}_RAW")
    set(_aliasRGroupPropName     "${_settingsPropPrefix}A${_flagOrGroup}_REMOVE_GROUP")
    set(_aliasAllowMultiPropName "${_settingsPropPrefix}A${_flagOrGroup}_ALLOW_MULTIPLE")

    # Removing aliases and splitting operations into remove group/add flag categories.
    if(_operation MATCHES "^SET$")
      list(FIND _aliases "${_flagOrGroup}" _aliasIdx)
      if(_aliasIdx LESS 0)
        list(APPEND _flagsToAdd "${_flagOrGroup}")
      else()
        get_property(_rawValueSet    GLOBAL PROPERTY "${_aliasRawPropName}" SET)
        get_property(_removeGroupSet GLOBAL PROPERTY "${_aliasRGroupPropName}" SET)
        get_property(_allowMultiple  GLOBAL PROPERTY "${_aliasAllowMultiPropName}")

        if(_removeGroupSet)
          get_property(_removeGroup GLOBAL PROPERTY "${_aliasRGroupPropName}")
          list(APPEND _groupsToRemove "${_removeGroup}")
        elseif(_rawValueSet)
          get_property(_rawValue    GLOBAL PROPERTY "${_aliasRawPropName}")
          list(APPEND _flagsToAdd "${_rawValue}")
        endif()
      endif()
    elseif(_operation MATCHES "^SET_RAW$")
      list(APPEND _flagsToAdd "${_flagOrGroup}")
    elseif(_operation MATCHES "^REMOVE_GROUP$")
      list(APPEND _groupsToRemove "$N{_flagOrGroup}")
    endif()

    # Taking into consideration mutual exclusion groups.
    if(NOT _allowMultiple)
      list(REMOVE_DUPLICATES _flagsToAdd)
    endif()
    
    foreach(_flagToAdd ${_flagsToAdd})
      string(REPLACE ";" "\;" _flagToAdd "${_flagToAdd}") # [WA#1] Must escape ; again if occurred in item.

      foreach(_group ${_groups})
        string(REPLACE ";" "\;" _group "${_group}") # [WA#1] Must escape ; again if occurred in item.

        set(_groupedFlagsPropName "${_settingsPropPrefix}G${_group}_FLAGS")

        get_property(_groupedFlags GLOBAL PROPERTY "${_groupedFlagsPropName}")
        foreach(_groupedFlag ${_groupedFlags})
          string(REPLACE ";" "\;" _groupedFlag "${_groupedFlag}") # [WA#1] Must escape ; again if occurred in item.

          if(_flagToAdd MATCHES "([ ]+|^)(${_groupedFlag})([ ]+|$)")
            list(APPEND _groupsToRemove "${_group}")
            break()
          endif()
        endforeach()
      endforeach()
    endforeach()

    # Removing all groups of mutually exclusive options that collide with added flags or
    # has been selected to remove.
    list(REMOVE_DUPLICATES _groupsToRemove)

    #message("GR ---> ${_groupsToRemove}")
    #message("FA ---> ${_flagsToAdd}")

    foreach(_groupToRemove ${_groupsToRemove})
      string(REPLACE ";" "\;" _groupToRemove "${_groupToRemove}") # [WA#1] Must escape ; again if occurred in item.

      set(_groupedFlagsPropName "${_settingsPropPrefix}G${_groupToRemove}_FLAGS")

      list(FIND _groups "${_groupToRemove}" _groupToRemoveIdx)
      if(_groupToRemoveIdx LESS 0)
        string(REGEX REPLACE "^N" "" _groupToRemove "${_groupToRemove}")
        message(WARNING "Group of options to remove \"${_groupToRemove}\" cannot be found and will be omitted.")
      else()
        get_property(_groupedFlags GLOBAL PROPERTY "${_groupedFlagsPropName}")
        foreach(_flagVarName ${_flagVarNames})
          string(REPLACE ";" "\;" _flagVarName "${_flagVarName}") # [WA#1] Must escape ; again if occurred in item.

          intel_flag_remove_re("_updatedFlags_${_flagVarName}" "${_groupedFlags}") # [WA#2] To handle ; correctly some lists must be put in string form.
        endforeach()
      endif()
    endforeach()

    # Adding flags.
    if(NOT _allowMultiple)
      # If multiple flags are not allowed, the flags must be moved to main variable.
      foreach(_flagVarName ${_flagVarNames})
        string(REPLACE ";" "\;" _flagVarName "${_flagVarName}") # [WA#1] Must escape ; again if occurred in item.

        if(NOT (_flagVarName STREQUAL flagsMainVarName))
          intel_flag_remove("_updatedFlags_${_flagVarName}" "${_flagsToAdd}") # [WA#2] To handle ; correctly some lists must be put in string form.
        endif()
      endforeach()
      intel_flag_add_once("${_updatedFlagsMainVarName}" "${_flagsToAdd}") # [WA#2] To handle ; correctly some lists must be put in string form.
    else()
      foreach(_flagToAdd ${_flagsToAdd})
        string(REPLACE ";" "\;" _flagToAdd "${_flagToAdd}") # [WA#1] Must escape ; again if occurred in item.
        set("${_updatedFlagsMainVarName}" "${${_updatedFlagsMainVarName}} ${_flagToAdd}")
      endforeach()
    endif()

    math(EXPR _operationIdx "${_operationIdx} + 1")
  endforeach()

  # Returning flags.
  foreach(_flagVarName ${_flagVarNames})
    string(REPLACE ";" "\;" _flagVarName "${_flagVarName}")
    set("${_flagVarName}" "${_updatedFlags_${_flagVarName}}" PARENT_SCOPE)
  endforeach()
endfunction()

# Applies settings to configuration flag variable (variable which has per-configuration subvariables).
# Settings are applied according to registered configuration (by intel_flag_register_settings).
#
# intel_config_flag_apply_settings(
#     settingsDomainName
#     flagsVarBaseName
#     { PATTERN | NEG_PATTERN | ALL_PATTERN | ALL_PATTERN_NOINHERIT }
#     configPattern
#     [SET          flag [flag [...]]]
#     [SET_RAW      flag [flag [...]]]
#     [REMOVE_GROUP groupName [groupName [...]]]
#     [{SET|SET_RAW|REMOVE_GROUP} ...]
#     [...]
#   )
#
# Allowed operation to apply:
# SET          - sets flag (takes into consideration aliases).
# SET_RAW      - sets flag (does NOT take aliases into consideration).
# REMOVE_GROUP - removes all flags identified by specified group.
# Operations are applied in definition order.
#
# @param settingsDomainName Domain name for settings / relations. The domain name allows to differentate and
#                           to create multiple sets of relations. All operations on settings functions use
#                           domain name to identify relations domain.
# @param flagsVarBaseName   Base name of flags variable. Per-configuration variables are constructed by
#                           attaching _<CONFIG> suffix. Base name variable is treated as variable
#                           which contains flags common to all configurations.
# @param patternType        Pattern type:
#                            - PATTERN     - Normal regular expression pattern (select configuration that match
#                                            pattern). Behaves like ALL_PATTERN when matched all configurations.
#                            - NEG_PATTERN - Negated regular expression pattern (selects configuration that do not
#                                            match pattern). Behaves like ALL_PATTERN when matched all configurations.
#                            - ALL_PATTERN           - configPattern parameter is ignored and all configurations
#                                                      are selected. The settings are applied to common/base configuration
#                                                      variable (and they are inherited).
#                                                      When inherited common flag is removed, the removal affects all configurations
#                                                      (flags removed from common variable due to redundance or mutual exclusion
#                                                      are not moved to specific configs on applied operations).
#                            - ALL_PATTERN_NOINHERIT - configPattern parameter is ignored and all configurations
#                                                      are selected. The settings are applied to all specific configuration
#                                                      variables (and they are NOT inherited). Use this if you want to
#                                                      preserve flag in configurations that are mutually exclusive with flags
#                                                      in other configurations.
# @param configPattern      Regular expression which select configurations to which settings will be applied.
# @param flag               Flags to set (SET or SET_RAW).
# @param groupName          Names of groups of flags to remove (REMOVE_GROUP).
function(intel_config_flag_apply_settings settingsDomainName flagsVarBaseName patternType configPattern)
  set(_updatedFlags "${${flagsVarBaseName}}")

  set(_negativePattern NO)
  set(_matchAllConfigs NO)
  set(_configNoinherit NO)

  if(patternType MATCHES "^PATTERN$")
    # Only for check.
  elseif(patternType MATCHES "^NEG_PATTERN$")
    set(_negativePattern YES)
  elseif(patternType MATCHES "^ALL_PATTERN$")
    set(_matchAllConfigs YES)
  elseif(patternType MATCHES "^ALL_PATTERN_NOINHERIT$")
    set(_matchAllConfigs YES)
    set(_configNoinherit YES)
  else()
    message(FATAL_ERROR "Pattern type \"${patternType}\" is invalid. Supported patter types/keywords:\nPATTERN, NEG_PATTERN, ALL_PATTERN, ALL_PATTERN_NOINHERIT")
  endif()


  set(_matchedAllConfigs     YES)
  set(_selectedConfigs       "")
  set(_selectedFlagsVarNames "")
  foreach(_configName ${CMAKE_CONFIGURATION_TYPES})
    string(REPLACE ";" "\;" _configName "${_configName}") # [WA#1] Must escape ; again if occurred in item.

    if(_matchAllConfigs OR (_negativePattern AND (NOT (_configName MATCHES "${configPattern}"))) OR ((NOT _negativePattern) AND (_configName MATCHES "${configPattern}")))
      string(TOUPPER "${_configName}" _upperConfigName)
      set(_updatedConfigFlagsName "_updatedFlags_${_upperConfigName}")

      set("${_updatedConfigFlagsName}" "${${flagsVarBaseName}_${_upperConfigName}}")

      list(APPEND _selectedConfigs "${_upperConfigName}")
      list(APPEND _selectedFlagsVarNames "${_updatedConfigFlagsName}")
    else()
      set(_matchedAllConfigs NO)
    endif()
  endforeach()

  if(_matchedAllConfigs AND (NOT _configNoinherit))
    intel_flag_apply_settings("${settingsDomainName}" _updatedFlags FLAG "${_selectedFlagsVarNames}" "${ARGN}") # [WA#2] To handle ; correctly some lists must be put in string form.
  else()
    foreach(_selectedFlagsVarName ${_selectedFlagsVarNames})
      string(REPLACE ";" "\;" _selectedFlagsVarName "${_selectedFlagsVarName}") # [WA#1] Must escape ; again if occurred in item.

      intel_flag_apply_settings("${settingsDomainName}" "${_selectedFlagsVarName}" FLAG _updatedFlags "${ARGN}") # [WA#2] To handle ; correctly some lists must be put in string form.
    endforeach()
  endif()


  set("${flagsVarBaseName}" "${_updatedFlags}" PARENT_SCOPE)
  foreach(_upperConfigName ${_selectedConfigs})
    string(REPLACE ";" "\;" _upperConfigName "${_upperConfigName}") # [WA#1] Must escape ; again if occurred in item.
    set(_updatedConfigFlagsName "_updatedFlags_${_upperConfigName}")

    set("${flagsVarBaseName}_${_upperConfigName}" "${${_updatedConfigFlagsName}}" PARENT_SCOPE)
  endforeach()
endfunction()

# ===================================== Host / Target Architecture =====================================

# Detects host and target architecture.
#
# Currently supports: Windows32, Windows64, WindowsARM, Android32, Android64, AndroidMIPS, AndroidARM,
#                     Linux32, Linux64, LinuxMIPS, LinuxARM, Darwin32, Darwin64, DarwinMIPS, DarwinARM.
#
# @param targetArchVarName Name of variable placeholder for target architecture.
# @param hostArchVarName   Name of variable placeholder for host architecture.
# @param targetArchBits    Hint about target architecture. (in case that the user specify the architecture explicitly)
function(intel_arch_detect targetArchVarName hostArchVarName targetArchBits)
  string(TOLOWER "${CMAKE_GENERATOR}" _cmakeGenerator)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _cmakeTargetProcessor)

  # Target architecture:
  # Detect target architecture on Windows using suffix from generator.
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(targetArchBits MATCHES 32)
      set(_targetArchitecture "Windows32")
    elseif(targetArchBits MATCHES 64)
      set(_targetArchitecture "Windows64")
    else()
      if(_cmakeGenerator MATCHES " (arm|aarch64)$")
        set(_targetArchitecture "WindowsARM")
      elseif(_cmakeGenerator MATCHES " (x64|x86_64|win64|intel64|amd64|64[ ]*\\-[ ]*bit)$")
        set(_targetArchitecture "Windows64")
      else()
        set(_targetArchitecture "Windows32")
      endif()
    endif()
  # Use system processor set by toolchain or CMake.
  elseif(ANDROID OR (CMAKE_SYSTEM_NAME MATCHES "Linux") OR (CMAKE_SYSTEM_NAME MATCHES "Darwin"))
    if(ANDROID)
      set(_targetArchOS "Android")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
      set(_targetArchOS "Linux")
    else()
      set(_targetArchOS "Darwin")
    endif()
    
    if(targetArchBits MATCHES 32)
      set(_targetArchitecture "${_targetArchOS}32")
    elseif(targetArchBits MATCHES 64)
      set(_targetArchitecture "${_targetArchOS}64")
    else()
      if(_cmakeTargetProcessor MATCHES "(x64|x86_64|amd64|64[ ]*\\-[ ]*bit)")
        set(_targetArchitecture "${_targetArchOS}64")
      elseif(_cmakeTargetProcessor MATCHES "(x32|x86|i[0-9]+86|32[ ]*\\-[ ]*bit)")
        set(_targetArchitecture "${_targetArchOS}32")
      elseif(_cmakeTargetProcessor MATCHES "mips")
        set(_targetArchitecture "${_targetArchOS}MIPS")
      else()
        set(_targetArchitecture "${_targetArchOS}ARM")
      endif()
    endif()
  else()
    set(_targetArchitecture "Unknown-NOTFOUND")
  endif()

  # Host architecture:
  # Detect system architecture using WMI.
  if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set(_osArchitecture "32-bit")
    execute_process(
        COMMAND powershell -NonInteractive -Command "(Get-WmiObject -Class Win32_OperatingSystem).OSArchitecture.ToLowerInvariant()"
        TIMEOUT 10
        OUTPUT_VARIABLE _osArchitecture
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

    if(NOT (_osArchitecture MATCHES "^(x64|x86_64|win64|intel64|amd64|64[ ]*\\-[ ]*bit)$"))
      set(_hostArchitecture "Windows32")
    else()
      set(_hostArchitecture "Windows64") # WindowsARM cannot be host.
    endif()
  # Use 'uname -m' to detect kernel architecture.
  elseif((CMAKE_HOST_SYSTEM_NAME MATCHES "Linux") OR (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin"))
    if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
      set(_hostArchOS "Linux")
    else()
      set(_hostArchOS "Darwin")
    endif()

    set(_osArchitecture "x86_64")
    execute_process(
        COMMAND uname -m
        TIMEOUT 10
        OUTPUT_VARIABLE _osArchitecture
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    string(TOLOWER "${_osArchitecture}" _osArchitecture)

    if(_osArchitecture MATCHES "(x64|x86_64|amd64|64[ ]*\\-[ ]*bit)")
      set(_hostArchitecture "${_hostArchOS}64")
    elseif(_osArchitecture MATCHES "(x32|x86|i[0-9]+86|32[ ]*\\-[ ]*bit)")
      set(_hostArchitecture "${_hostArchOS}32")
    elseif(_osArchitecture MATCHES "mips")
      set(_hostArchitecture "${_hostArchOS}MIPS")
    else()
      set(_hostArchitecture "${_hostArchOS}ARM")
    endif()
  else()
    set(_hostArchitecture "Unknown-NOTFOUND")
  endif()

  set("${targetArchVarName}" "${_targetArchitecture}" PARENT_SCOPE)
  set("${hostArchVarName}"   "${_hostArchitecture}" PARENT_SCOPE)
endfunction()


# Determines whether architecture is valid (accepts only normalized).
#
# @param retValName Name of variable placeholder where result will be returned.
# @param arch       Architecture name to validate. Architecture should be normalized.
function(intel_arch_validate retVarName arch)
  # Allowed architectures (list).
  set(__allowedArchs
      "Windows32" "Windows64"               "WindowsARM"
      "Android32" "Android64" "AndroidMIPS" "AndroidARM"
      "Linux32"   "Linux64"   "LinuxMIPS"   "LinuxARM"
      "Darwin32"  "Darwin64"  "DarwinMIPS"  "DarwinARM"
    )

  list(FIND __allowedArchs "${arch}" _allowedArchIdx)
  if(_allowedArchIdx LESS 0)
    set("${retVarName}" NO  PARENT_SCOPE)
  else()
    set("${retVarName}" YES PARENT_SCOPE)
  endif()
endfunction()


# Normalizes architecture name. If architecture is not supported by helper functions
# the "Unknown-NOTFOUND" will be returned.
#
# Currently supports: Windows32, Windows64, WindowsARM, Android32, Android64, AndroidMIPS, AndroidARM,
#                     Linux32, Linux64, LinuxMIPS, LinuxARM, Darwin32, Darwin64, DarwinMIPS, DarwinARM.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param arch       Architecture name to normalize / filter.
function(intel_arch_normalize retVarName arch)
  string(TOLOWER "${arch}" _arch)
  string(STRIP "${_arch}" _arch)

  if(_arch MATCHES "^win")
    set(_osPart "Windows")
  elseif(_arch MATCHES "^and")
    set(_osPart "Android")
  elseif(_arch MATCHES "^lin")
    set(_osPart "Linux")
  elseif(_arch MATCHES "^(mac|app|dar)")
    set(_osPart "Darwin")
  else()
    set(_osPart "Windows")
  endif()

  if(_arch MATCHES "64$")
    set(_cpuPart "64")
  elseif(_arch MATCHES "(32|86)$")
    set(_cpuPart "32")
  elseif(_arch MATCHES "mips^")
    set(_cpuPart "MIPS")
  elseif(_arch MATCHES "arm|aarch64")
    set(_cpuPart "ARM")
  else()
    set("${retVarName}" "Unknown-NOTFOUND" PARENT_SCOPE)
    return()
  endif()

  set(_normalizedArch "${_osPart}${_cpuPart}")
  intel_arch_validate(_archValid "${_normalizedArch}")
  if(_archValid)
    set("${retVarName}" "${_normalizedArch}" PARENT_SCOPE)
  else()
    set("${retVarName}" "Unknown-NOTFOUND" PARENT_SCOPE)
  endif()
endfunction()


# Gets OS platform used in specified architecture. If it cannot be determined
# the "Unknown-NOTFOUND" will be returned.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param arch       Architecture name to get info from.
function(intel_arch_get_os retVarName arch)
  if(arch MATCHES "^(Windows|Android|Linux|Darwin)(.*)$")
    set("${retVarName}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
  else()
    set("${retVarName}" "Unknown-NOTFOUND" PARENT_SCOPE)
  endif()
endfunction()


# Gets CPU platform used in specified architecture. If it cannot be determined
# the "Unknown-NOTFOUND" will be returned.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param arch       Architecture name to get info from.
function(intel_arch_get_cpu retVarName arch)
  if(arch MATCHES "^(Windows|Android|Linux|Darwin)(.*)$")
    set("${retVarName}" "${CMAKE_MATCH_2}" PARENT_SCOPE)
  else()
    set("${retVarName}" "Unknown-NOTFOUND" PARENT_SCOPE)
  endif()
endfunction()


# Determines whether cross-compilation is needed.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param targetArch Target architecture (it will be normalized).
# @param hostArch   Host architecture (it will be normalized).
function(intel_arch_crosscompile_needed retVarName targetArch hostArch)
  # Allowed cross-executions (keys list, lists for each key).
  # Key:    host architecture.
  # Values: target architecture that can be run on host (different than host).
  set(__allowedCrossExecution "Windows64")
  set(__allowedCrossExecution_Windows64 "Windows32")

  intel_arch_normalize(_targetArch "${targetArch}")
  intel_arch_normalize(_hostArch   "${hostArch}")

  if(_targetArch STREQUAL _hostArch)
    set("${retVarName}" NO PARENT_SCOPE)
  else()
    list(FIND __allowedCrossExecution "${_hostArch}" _keyIdx)
    if(_keyIdx LESS 0)
      set("${retVarName}" YES PARENT_SCOPE)
    else()
      list(FIND "__allowedCrossExecution_${_hostArch}" "${_targetArch}" _valIdx)
      if(_valIdx LESS 0)
        set("${retVarName}" YES PARENT_SCOPE)
      else()
        set("${retVarName}" NO PARENT_SCOPE)
      endif()
    endif()
  endif()
endfunction()

# ================================== Tools / custom build step helpers =================================

# Helper function which register tool activity and allow to generate command-line for it.
#
# intel_tool_register_activity(
#                          targetName
#                          activityName
#     [ARCH_OS             archOs
#       [ARCH_CPU          archCpu]]
#     [PLACEHOLDER         placeholder]
#     [ALLOW_EXTRA_PARAMS  allowExtraParams]
#                          [commandElem [commandElem [...]]]
#   )
#
# @param targetName       Target name of imported / created tool which will be executed during activity.
# @param activityName     Name of registered activity. The name can be registered multiple times (for different architectures).
# @param archOs           OS architecture compatible with activity. If not specified, the activity is treated as OS-generic.
# @param archCpu          CPU architecture compatible with activity. If not specified, the activity is treated as CPU-generic.
#                         Can be only specified when OS architecture is specified.
# @param placeholder      Text that will be searched to replace with actual parameters.
# @param allowExtraParams If TRUE the additional commandline parameters specified in intel_tool_get_activity_commandline will
#                         be passed to command-line (templated as last commandElem).
# @param commandElem      Command-line elements (with optional placeholders for parameters).
function(intel_tool_register_activity targetName activityName)
  if(NOT (TARGET "${targetName}"))
    message(FATAL_ERROR "Target \"${targetName}\" does not exist. Activity \"${activityName}\" will not be registered.")
  endif()
  if(activityName MATCHES "^[ ]*$")
    message(FATAL_ERROR "Activity name is empty or blank.")
  endif()


  set(_archOs    "")
  set(_archOsSet NO)
  set(_archCpu    "")
  set(_archCpuSet NO)
  set(_placeholder "<<<param>>>")
  set(_allowExtraParams NO)
  set(_commandElems "")

  set(_parseState 0)
  foreach(_toolArg ${ARGN})
    string(REPLACE ";" "\;" _toolArg "${_toolArg}") # [WA#1] Must escape ; again if occurred in item.

    # States: [0] *1( "ARCH_OS" [1] <param> [2] *1( "ARCH_CPU" [3] <param> [4] ) ) *1( "PLACEHOLDER" [5] <param> [6] ) *1( "ALLOW_EXTRA_PARAMS" [7] <param> [8] ) *<param> [8]
    # Transitions: 0 (ARCH_OS) -> 1 -> 2
    #              0 (PLACEHOLDER) -> 5 -> 6
    #              0 (ALLOW_EXTRA_PARAMS) -> 7 -> 8
    #              0 -> 8
    #              2 (ARCH_CPU) -> 3 -> 4
    #              2 (PLACEHOLDER) -> 5 -> 6
    #              2 (ALLOW_EXTRA_PARAMS) -> 7 -> 8
    #              2 -> 8
    #              4 (PLACEHOLDER) -> 5 -> 6
    #              4 (ALLOW_EXTRA_PARAMS) -> 7 -> 8
    #              4 -> 8
    #              6 (ALLOW_EXTRA_PARAMS) -> 7 -> 8
    #              6 -> 8
    # Stop States: 8
    if(_parseState EQUAL 0)
      if(_toolArg MATCHES "^ARCH_OS$")
        set(_parseState 1)
      elseif(_toolArg MATCHES "^PLACEHOLDER$")
        set(_parseState 5)
      elseif(_toolArg MATCHES "^ALLOW_EXTRA_PARAMS$")
        set(_parseState 7)
      else()
        list(APPEND _commandElems "${_toolArg}")
        set(_parseState 8)
      endif()
    elseif(_parseState EQUAL 1)
      set(_archOs    "${_toolArg}")
      set(_archOsSet YES)
      set(_parseState 2)
    elseif(_parseState EQUAL 2)
      if(_toolArg MATCHES "^ARCH_CPU$")
        set(_parseState 3)
      elseif(_toolArg MATCHES "^PLACEHOLDER$")
        set(_parseState 5)
      elseif(_toolArg MATCHES "^ALLOW_EXTRA_PARAMS$")
        set(_parseState 7)
      else()
        list(APPEND _commandElems "${_toolArg}")
        set(_parseState 8)
      endif()
    elseif(_parseState EQUAL 3)
      set(_archCpu    "${_toolArg}")
      set(_archCpuSet YES)
      set(_parseState 4)
    elseif(_parseState EQUAL 4)
      if(_toolArg MATCHES "^PLACEHOLDER$")
        set(_parseState 5)
      elseif(_toolArg MATCHES "^ALLOW_EXTRA_PARAMS$")
        set(_parseState 7)
      else()
        list(APPEND _commandElems "${_toolArg}")
        set(_parseState 8)
      endif()
    elseif(_parseState EQUAL 5)
      set(_placeholder "${_toolArg}")
      set(_parseState 6)
    elseif(_parseState EQUAL 6)
      if(_toolArg MATCHES "^ALLOW_EXTRA_PARAMS$")
        set(_parseState 7)
      else()
        list(APPEND _commandElems "${_toolArg}")
        set(_parseState 8)
      endif()
    elseif(_parseState EQUAL 7)
      set(_allowExtraParams "${_toolArg}")
      set(_parseState 8)
    elseif(_parseState EQUAL 8)
      list(APPEND _commandElems "${_toolArg}")
    else()
      message(FATAL_ERROR "Invalid parameter token near \"${_toolArg}\".")
    endif()
  endforeach()
  if(NOT (_parseState EQUAL 8))
    message(FATAL_ERROR "Invalid number of parameters.")
  endif()

  intel_regex_escape(_placeholderRe "${_placeholder}")
  string(LENGTH "${_placeholderRe}" _placeholderReLen)
  if(_placeholderReLen EQUAL 0)
    message(FATAL_ERROR "Placeholder cannot be empty string.")
  endif()


  string(REPLACE ";" "\;" _activityName "${activityName}")
  if(_archCpuSet)
    set(_activityPropName "INTEL_CMAKE_PROPERTY__TOOL__OS${_archOs}_CPU${_archCpu}_A${_activityName}__ACTIVITY")
  elseif(_archOsSet)
    set(_activityPropName "INTEL_CMAKE_PROPERTY__TOOL__OS${_archOs}_A${_activityName}__ACTIVITY")
  else()
    set(_activityPropName "INTEL_CMAKE_PROPERTY__TOOL__A${_activityName}__ACTIVITY")
  endif()

  set(_commandElemIdx 0)
  foreach(_commandElem ${_commandElems})
    string(REPLACE ";" "\;" _commandElem "${_commandElem}") # [WA#1] Must escape ; again if occurred in item.

    set(_commandElemParts "")
    set(_scannedElem "${_commandElem}")
    while(TRUE)
      if(_scannedElem MATCHES "${_placeholderRe}(.*)$")
        set(_scannedElemRest "${CMAKE_MATCH_1}")
        string(REGEX REPLACE "${_placeholderRe}.*$" "" _scannedElemPrefix "${_scannedElem}")
        list(APPEND _commandElemParts "P${_scannedElemPrefix}")
        set(_scannedElem "${_scannedElemRest}")
      else()
        list(APPEND _commandElemParts "P${_scannedElem}")
        break()
      endif()
    endwhile()

    set_property(TARGET "${targetName}" PROPERTY "${_activityPropName}_I${_commandElemIdx}" "${_commandElemParts}")

    math(EXPR _commandElemIdx "${_commandElemIdx} + 1")
  endforeach()
  set_property(TARGET "${targetName}" PROPERTY "${_activityPropName}"       ${_commandElemIdx})
  set_property(TARGET "${targetName}" PROPERTY "${_activityPropName}_EXTRA" ${_allowExtraParams})
endfunction()

# Gets command-line for selected activity for specified tool target. Takes host architecture
# into consideration - uses INTEL_CMAKE_OPTION__ARCHITECTURE_HOST variable do determine it.
#
# Allows to unify usage of tools between multiple platforms. The activity must be
# registered before it can be used. The activity can be registered using
# intel_tool_register_activity function, for example:
#
# add_executable(sampleTool IMPORTED)
# set_property(TARGET sampleTool PROPERTY IMPORTED_LOCATION "<Host OS-specific path to tool>")
# intel_tool_register_activity(sampleTool sampleOperation
#     ARCH_OS Windows
#     /sampleOption /sampleWindowsOption=<<<param>>>
#   )
# intel_tool_register_activity(sampleTool sampleOperation
#     ARCH_OS  Linux
#     ARCH_CPU 64
#     -sampleOption -sampleLinuxOption=<<<param>>>
#   )
#
# will register simpleOperation activity for sampleTool. Returned command-line will depend
# on host architecture:
#
# intel_tool_get_activity_commandline(commandLine sampleTool sampleOperation param1Value)
#
# will write to commandLine:
#   [Windows] "sampleTool" "/sampleOption" "/sampleWindowsOption=param1Value
#   [Linux64] "sampleTool" "-sampleOption" "-sampleLinuxOption=param1Value
#
# @param retVarName    Name of variable where command-line text will be returned.
#                      The list compatible with COMMAND from custom build target/command
#                      is returned. It should not be used outside this element.
# @param targetName    Target name of imported / created tool which will be executed during activity.
#                      The activity must be registered for tool target name.
# @param activityName  Name of activity for which command-line will be returned. The function
#                      throws fatal error when activity cannot be found for current tool target
#                      and current host architecture (read from INTEL_CMAKE_OPTION__ARCHITECTURE_HOST).
# @param [param [...]] Additional configurable parameters for command-line.
#                      Values specified here fill the placeholder in registered command-line elements.
function(intel_tool_get_activity_commandline retVarName targetName activityName)
  # Name of variable where current host architecture is stored.
  set(__hostArchVarName "INTEL_CMAKE_OPTION__ARCHITECTURE_HOST")

  if(NOT (TARGET "${targetName}"))
    message(FATAL_ERROR "Target \"${targetName}\" does not exist. The tool is not available.")
  endif()
  if(activityName MATCHES "^[ ]*$")
    message(FATAL_ERROR "Activity name is empty or blank.")
  endif()


  # Select most matching activity (with most matching architecture - most specific first, most generic last).
  intel_arch_get_os(_archOs   "${${__hostArchVarName}}")
  intel_arch_get_cpu(_archCpu "${${__hostArchVarName}}")

  string(REPLACE ";" "\;" _activityName "${activityName}")
  set(_activityPropNames
      "INTEL_CMAKE_PROPERTY__TOOL__OS${_archOs}_CPU${_archCpu}_A${_activityName}__ACTIVITY"
      "INTEL_CMAKE_PROPERTY__TOOL__OS${_archOs}_A${_activityName}__ACTIVITY"
      "INTEL_CMAKE_PROPERTY__TOOL__A${_activityName}__ACTIVITY"
    )

  set(_hasActivity NO)
  foreach(_activityPropName ${_activityPropNames})
    string(REPLACE ";" "\;" _activityPropName "${_activityPropName}") # [WA#1] Must escape ; again if occurred in item.

    get_property(_hasActivity TARGET "${targetName}" PROPERTY "${_activityPropName}" SET)
    if(_hasActivity)
      set(_selectedActivityPropName "${_activityPropName}")
      break()
    endif()
  endforeach()

  if(NOT _hasActivity)
    message(FATAL_ERROR "The tool idenfified by target \"${targetName}\" does not support activity \"${activityName}\" for current architecture:\n${_archOs}, ${_archCpu}.")
  endif()


  # Template all placeholders.
  list(LENGTH ARGN _paramsCount)

  get_property(_commandElemsCount TARGET "${targetName}" PROPERTY "${_selectedActivityPropName}")
  get_property(_allowExtraParams  TARGET "${targetName}" PROPERTY "${_selectedActivityPropName}_EXTRA")

  set(_currentParamIdx 0)
  set(_commandLine "${targetName}")
  if(_commandElemsCount GREATER 0)
    math(EXPR _commandElemsCountM1 "${_commandElemsCount} - 1")
    foreach(_commandElemIdx RANGE 0 ${_commandElemsCountM1})
      get_property(_commandElemParts TARGET "${targetName}" PROPERTY "${_selectedActivityPropName}_I${_commandElemIdx}")

      # Support for templating extra parameters by using last template sufficient number of times.
      if(_allowExtraParams AND (_commandElemIdx EQUAL _commandElemsCountM1))
        if(_currentParamIdx LESS _paramsCount)
          math(EXPR _maxIterations "${_paramsCount} - ${_currentParamIdx}")
        else()
          set(_maxIterations 0)
        endif()
      else()
        set(_maxIterations 1)
      endif()

      while(_maxIterations GREATER 0)
        set(_firstPart YES)
        foreach(_commandElemPart ${_commandElemParts})
          string(REPLACE ";" "\;" _commandElemPart "${_commandElemPart}") # [WA#1] Must escape ; again if occurred in item.

          string(REGEX REPLACE "^P" "" _commandElemPart "${_commandElemPart}")
          if(_firstPart)
            set(_commandElem "${_commandElemPart}")
            set(_firstPart NO)
          else()
            if(_currentParamIdx LESS _paramsCount)
              list(GET ARGN ${_currentParamIdx} _paramValue)
              string(REPLACE ";" "\;" _paramValue "${_paramValue}") # [WA#1] Must escape ; again if occurred in item.
              math(EXPR _currentParamIdx "${_currentParamIdx} + 1")
            else()
              set(_paramValue "")
            endif()
            set(_commandElem "${_commandElem}${_paramValue}${_commandElemPart}")
          endif()
        endforeach()
        list(APPEND _commandLine "${_commandElem}")

        # Finish templating extra parameters when there is no extra parameters
        # or number of repeats is equal number of extra parameters (when template
        # does not have placeholder).
        if(_currentParamIdx LESS _paramsCount)
          math(EXPR _maxIterations "${_maxIterations} - 1")
        else()
          set(_maxIterations 0)
        endif()
      endwhile()
    endforeach()
  elseif(_allowExtraParams)
    while(_currentParamIdx LESS _paramsCount)
      list(GET ARGN ${_currentParamIdx} _paramValue)
      string(REPLACE ";" "\;" _paramValue "${_paramValue}") # [WA#1] Must escape ; again if occurred in item.
      math(EXPR _currentParamIdx "${_currentParamIdx} + 1")
      list(APPEND _commandLine "${_paramValue}")
    endwhile()
  endif()


  set("${retVarName}" "${_commandLine}" PARENT_SCOPE)
endfunction()

# ======================================== Custom configuration ========================================

# Adds custom build configuration settings.
#
# @param configName     Name of new custom configuration.
# @param copyConfigName Name of configuration which new configuration will settings will be copied from.
# @param [configPath]   Path element that represents custom configuration.
function(intel_custom_build_add configName copyConfigName)
  string(TOUPPER "${configName}" _configName)
  string(TOUPPER "${copyConfigName}" _copyConfigName)

  set("CMAKE_CXX_FLAGS_${_configName}"           "${CMAKE_CXX_FLAGS_${_copyConfigName}}"           CACHE STRING "C++ compilder flags used during \"${configName}\" builds.")
  set("CMAKE_C_FLAGS_${_configName}"             "${CMAKE_C_FLAGS_${_copyConfigName}}"             CACHE STRING "Flags used during \"${configName}\" builds.")
  set("CMAKE_EXE_LINKER_FLAGS_${_configName}"    "${CMAKE_EXE_LINKER_FLAGS_${_copyConfigName}}"    CACHE STRING "Linker flags used during \"${configName}\" builds for executables.")
  set("CMAKE_MODULE_LINKER_FLAGS_${_configName}" "${CMAKE_MODULE_LINKER_FLAGS_${_copyConfigName}}" CACHE STRING "Linker flags used during \"${configName}\" builds for modules.")
  set("CMAKE_SHARED_LINKER_FLAGS_${_configName}" "${CMAKE_SHARED_LINKER_FLAGS_${_copyConfigName}}" CACHE STRING "Linker flags used during \"${configName}\" builds for shared libraries.")
  set("CMAKE_STATIC_LINKER_FLAGS_${_configName}" "${CMAKE_STATIC_LINKER_FLAGS_${_copyConfigName}}" CACHE STRING "Linker flags used during \"${configName}\" builds for static libraries.")
  mark_as_advanced(
      "CMAKE_CXX_FLAGS_${_configName}"
      "CMAKE_C_FLAGS_${_configName}"
      "CMAKE_EXE_LINKER_FLAGS_${_configName}"
      "CMAKE_MODULE_LINKER_FLAGS_${_configName}"
      "CMAKE_SHARED_LINKER_FLAGS_${_configName}"
      "CMAKE_STATIC_LINKER_FLAGS_${_configName}"
    )

  list(LENGTH ARGN _optArgsCount)
  if(_optArgsCount GREATER 0)
    list(GET ARGN 0 _configPath)
    string(REPLACE ";" "\;" _configPath "${_configPath}") # [WA#1] Must escape ; again if occurred in item.
    set("INTEL_CMAKE_BUILD__${_configName}_CFG_PATH" "${_configPath}" PARENT_SCOPE)
  endif()
endfunction()


# Gets configuration path element corresponding to selected configuration.
#
# @param retValName Name of variable placeholder where result will be returned.
# @param configName Name of configuration.
function(intel_get_cfg_path retVarName configName)
  string(TOUPPER "${configName}" _configName)
  set(_configPathVarName "INTEL_CMAKE_BUILD__${_configName}_CFG_PATH")
  if(DEFINED "${_configPathVarName}")
    set("${retVarName}" "${${_configPathVarName}}" PARENT_SCOPE)
  else()
    set("${retVarName}" "${configName}" PARENT_SCOPE)
  endif()
endfunction()

# =========================================== Resource files ===========================================

# Registers configuration for resource files for specific idenitifer.
#
# The configuration of resource files can be fetched using intel_rc_get_resource() function.
#
# Current version of CMake does not allow to modify sources list after calling add_*()
# functions, so resources still need to be added by using intel_rc_get_resource() to get
# list of resource files and adding it to add_*().
#
# intel_rc_register_resource(
#     resourceId
#     [RESOURCES                      [importedId [importedId [...]]]]
#     [FILE                           rcFile [rcFile [...]]
#       [[APPEND] INCLUDE_DIRECTORIES [incudeDir [includeDir [...]]]]
#       [[APPEND] DEFINES             [define [define [...]]]]
#       [[APPEND] FLAGS               [flag [flag [...]]]]]
#   )
#
# @param resourceId Identifier of resource file or of group of resource files.
# @param importedId Identifiers of additional resources that should be added with currently registered
#                   resource.
# @param rcFile     Resource files which are be connected to specified resource identifier.
# @param incudeDir  Additional include directories for resource files.
# @param define     Additional defines for resource files.
# @param flag       Additional flags for resource files.
function(intel_rc_register_resource resourceId)
  if(NOT ("${resourceId}" MATCHES "[a-zA-Z0-9_]+"))
    message(FATAL_ERROR "The resource identifier \"${resourceId}\" is invalid.")
  endif()
  
  set(_rcPropPrefix "INTEL_CMAKE_PROPERTY__RC_RESOURCES_R${resourceId}_")
  set(_rcIdxPropName "${_rcPropPrefix}INDEX")

  # Preserve resource index to support multiple registration calls.
  get_property(_rcIdxSet GLOBAL PROPERTY "${_rcIdxPropName}" SET)
  if(_rcIdxSet)
    get_property(_rcIdx GLOBAL PROPERTY "${_rcIdxPropName}")
  else()
    set(_rcIdx 0)
  endif()

  set(_rcFilesPropName           "${_rcPropPrefix}G${_rcIdx}_FILES")
  set(_includeDirsPropName       "${_rcPropPrefix}G${_rcIdx}_INCDIRS")
  set(_definesPropName           "${_rcPropPrefix}G${_rcIdx}_DEFINES")
  set(_flagsPropName             "${_rcPropPrefix}G${_rcIdx}_FLAGS")
  set(_importResIdsPropName      "${_rcPropPrefix}G${_rcIdx}_IMPORTED_RES_IDS")
  set(_appendIncludeDirsPropName "${_includeDirsPropName}_APPEND")
  set(_appendDefinesPropName     "${_definesPropName}_APPEND")
  set(_appendFlagsPropName       "${_flagsPropName}_APPEND")

  set(_importResIds      "")
  set(_rcFiles           "")
  set(_includeDirs       "")
  set(_defines           "")
  set(_flags             "")
  set(_appendIncludeDirs NO)
  set(_appendDefines     NO)
  set(_appendFlags       NO)
  
  set(_parseState 1)
  foreach(_resArg ${ARGN})
    string(REPLACE ";" "\;" _resArg "${_resArg}") # [WA#1] Must escape ; again if occurred in item.

    # States: [0] <param> [1] *1( "RESOURCES" [100] *<param> [100] ) *1( "FILE" [2] <param> [3] *<param> [3]
    #         *1( *1( "APPEND" [4] ) "INCLUDE_DIRECTORIES" [5] *<param> [5] ) *1( *1( "APPEND" [6] ) "DEFINES" [7] *<param> [7] )
    #         *1( *1( "APPEND" [8] ) "FLAGS" [9] *<param> [9] ) )
    # Transitions: 0 -> 1 // by explict parameters
    #              1 (FILE) -> 2 -> 3
    #              1 (RESOURCES) -> 100
    #              3 (APPEND) -> 4 {6,8}
    #              3 (INCLUDE_DIRECTORIES) -> 5
    #              3 (DEFINES) -> 7
    #              3 (FLAGS) -> 9
    #              3 -> 3
    #              4 (INCLUDE_DIRECTORIES) -> 5
    #              4 (DEFINES) -> 7
    #              4 (FLAGS) -> 9
    #              5 (APPEND) -> 6 {8}
    #              5 (DEFINES) -> 7
    #              5 (FLAGS) -> 9
    #              5 -> 5
    #              6 (DEFINES) -> 7
    #              6 (FLAGS) -> 9
    #              7 (APPEND) -> 8
    #              7 (FLAGS) -> 9
    #              7 -> 7
    #              8 (FLAGS) -> 9
    #              9 -> 9
    #              100 (FILE) -> 2 -> 3
    #              100 -> 100
    # Stop States: 3, 5, 7, 9, 100
    if(_parseState EQUAL 1)
      if(_resArg MATCHES "^FILE$")
        set(_parseState 2)
      elseif(_resArg MATCHES "^RESOURCES$")
        set(_parseState 100)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_resArg}\".")
      endif()
    elseif(_parseState EQUAL 2)
      set(_rcFiles "${_resArg}")
      set(_parseState 3)
    elseif(_parseState EQUAL 3)
      if(_resArg MATCHES "^APPEND$")
        set(_parseState 4)
      elseif(_resArg MATCHES "^INCLUDE_DIRECTORIES$")
        set(_parseState 5)
      elseif(_resArg MATCHES "^DEFINES$")
        set(_parseState 7)
      elseif(_resArg MATCHES "^FLAGS$")
        set(_parseState 9)
      else()
        list(APPEND _rcFiles "${_resArg}")
      endif()
    elseif(_parseState EQUAL 4)
      if(_resArg MATCHES "^INCLUDE_DIRECTORIES$")
        set(_appendIncludeDirs YES)
        set(_parseState 5)
      elseif(_resArg MATCHES "^DEFINES$")
        set(_appendDefines YES)
        set(_parseState 7)
      elseif(_resArg MATCHES "^FLAGS$")
        set(_appendFlags YES)
        set(_parseState 9)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_resArg}\".")
      endif()
    elseif(_parseState EQUAL 5)
      if(_resArg MATCHES "^APPEND$")
        set(_parseState 6)
      elseif(_resArg MATCHES "^DEFINES$")
        set(_parseState 7)
      elseif(_resArg MATCHES "^FLAGS$")
        set(_parseState 9)
      else()
        list(APPEND _includeDirs "${_resArg}")
      endif()
    elseif(_parseState EQUAL 6)
      if(_resArg MATCHES "^DEFINES$")
        set(_appendDefines YES)
        set(_parseState 7)
      elseif(_resArg MATCHES "^FLAGS$")
        set(_appendFlags YES)
        set(_parseState 9)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_resArg}\".")
      endif()
    elseif(_parseState EQUAL 7)
      if(_resArg MATCHES "^APPEND$")
        set(_parseState 8)
      elseif(_resArg MATCHES "^FLAGS$")
        set(_parseState 9)
      else()
        list(APPEND _defines "${_resArg}")
      endif()
    elseif(_parseState EQUAL 8)
      if(_resArg MATCHES "^FLAGS$")
        set(_appendFlags YES)
        set(_parseState 9)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_resArg}\".")
      endif()
    elseif(_parseState EQUAL 9)
      list(APPEND _flags "${_resArg}")
    elseif(_parseState EQUAL 100)
      if(_resArg MATCHES "^FILE$")
        set(_parseState 2)
      else()
        list(APPEND _importResIds "${_resArg}")
      endif()
    else()
      message(FATAL_ERROR "Invalid parameter token near \"${_resArg}\".")
    endif()
  endforeach()
  if(NOT ((_parseState EQUAL 3) OR (_parseState EQUAL 5) OR (_parseState EQUAL 7) OR (_parseState EQUAL 9) OR (_parseState EQUAL 100)))
    message(FATAL_ERROR "Invalid number of parameters.")
  endif()

  list(REMOVE_DUPLICATES _rcFiles)
  list(REMOVE_DUPLICATES _importResIds)

  set_property(GLOBAL PROPERTY "${_rcFilesPropName}"           "${_rcFiles}")
  set_property(GLOBAL PROPERTY "${_includeDirsPropName}"       "${_includeDirs}")
  set_property(GLOBAL PROPERTY "${_definesPropName}"           "${_defines}")
  set_property(GLOBAL PROPERTY "${_flagsPropName}"             "${_flags}")
  set_property(GLOBAL PROPERTY "${_importResIdsPropName}"      "${_importResIds}")
  set_property(GLOBAL PROPERTY "${_appendIncludeDirsPropName}" "${_appendIncludeDirs}")
  set_property(GLOBAL PROPERTY "${_appendDefinesPropName}"     "${_appendDefines}")
  set_property(GLOBAL PROPERTY "${_appendFlagsPropName}"       "${_appendFlags}")

  math(EXPR _rcIdx "${_rcIdx} + 1")
  set_property(GLOBAL PROPERTY "${_rcIdxPropName}" "${_rcIdx}")
endfunction()


# Preconfigures and returns names of resource files connected to specified resource identifier.
#
# Preconfiguration sets additional definitions, include directories, etc. for each resource file.
# The returned files can be used only in current directory targets.
#
# @param rcFilesVarName Name of variable placeholder where names of resource files will be returned.
# @param resourceId     Idenfifier of resource file or of group of resource files.
function(intel_rc_get_resource rcFilesVarName resourceId)
  if(NOT ("${resourceId}" MATCHES "[a-zA-Z0-9_]+"))
    message(FATAL_ERROR "The resource identifier \"${resourceId}\" is invalid.")
  endif()

  set(_rcPropPrefix "INTEL_CMAKE_PROPERTY__RC_RESOURCES_R${resourceId}_")
  set(_rcIdxPropName "${_rcPropPrefix}INDEX")

  # Preserve resource index to support multiple registration calls.
  get_property(_rcIdxSet GLOBAL PROPERTY "${_rcIdxPropName}" SET)
  if(_rcIdxSet)
    get_property(_rcIdx GLOBAL PROPERTY "${_rcIdxPropName}")
  else()
    message(AUTHOR_WARNING "Resource connected to \"${resourceId}\" not found.")
    set(_rcIdx 0)
  endif()

  set(_targetRcFiles "")

  if(_rcIdx GREATER 0)
    math(EXPR _rcIdxM1 "${_rcIdx} - 1")

    foreach(_idx RANGE 0 "${_rcIdxM1}")
      set(_rcFilesPropName           "${_rcPropPrefix}G${_idx}_FILES")
      set(_includeDirsPropName       "${_rcPropPrefix}G${_idx}_INCDIRS")
      set(_definesPropName           "${_rcPropPrefix}G${_idx}_DEFINES")
      set(_flagsPropName             "${_rcPropPrefix}G${_idx}_FLAGS")
      set(_importResIdsPropName      "${_rcPropPrefix}G${_idx}_IMPORTED_RES_IDS")
      set(_appendIncludeDirsPropName "${_includeDirsPropName}_APPEND")
      set(_appendDefinesPropName     "${_definesPropName}_APPEND")
      set(_appendFlagsPropName       "${_flagsPropName}_APPEND")

      get_property(_rcFiles           GLOBAL PROPERTY "${_rcFilesPropName}")
      get_property(_includeDirs       GLOBAL PROPERTY "${_includeDirsPropName}")
      get_property(_defines           GLOBAL PROPERTY "${_definesPropName}")
      get_property(_flags             GLOBAL PROPERTY "${_flagsPropName}")
      get_property(_importResIds      GLOBAL PROPERTY "${_importResIdsPropName}")
      get_property(_appendIncludeDirs GLOBAL PROPERTY "${_appendIncludeDirsPropName}")
      get_property(_appendDefines     GLOBAL PROPERTY "${_appendDefinesPropName}")
      get_property(_appendFlags       GLOBAL PROPERTY "${_appendFlagsPropName}")

      set(_includeDirFlags "")
      foreach(_includeDir ${_includeDirs})
        string(REPLACE ";" "\;" _includeDir "${_includeDir}") # [WA#1] Must escape ; again if occurred in item.
        if(NOT (_includeDirFlags MATCHES "^$"))
          set(_includeDirFlags "${_includeDirFlags} /I\"${_includeDir}\"")
        else()
          set(_includeDirFlags "/I\"${_includeDir}\"")
        endif()
      endforeach()

      set(_flagFlags "")
      foreach(_flag ${_flags})
        string(REPLACE ";" "\;" _flag "${_flag}") # [WA#1] Must escape ; again if occurred in item.
        if(NOT (_flagFlags MATCHES "^$"))
          set(_flagFlags "${_flagFlags} ${_flag}")
        else()
          set(_flagFlags "${_flag}")
        endif()
      endforeach()

      if(NOT (_flagFlags MATCHES "^$"))
        set(_allFlags "${_flagFlags} ${_includeDirFlags}")
      else()
        set(_allFlags "${_includeDirFlags}")
      endif()

      foreach(_rcFile ${_rcFiles})
        string(REPLACE ";" "\;" _rcFile "${_rcFile}") # [WA#1] Must escape ; again if occurred in item.

        if(_appendDefines)
          set_property(SOURCE "${_rcFile}" APPEND PROPERTY COMPILE_DEFINITIONS "${_defines}")
        else()
          set_property(SOURCE "${_rcFile}" PROPERTY COMPILE_DEFINITIONS "${_defines}")
        endif()

        # NOTE: Currently there is no include dirs (per source), so appending options are limited.
        if(_appendFlags OR _appendIncludeDirs)
          set_property(SOURCE "${_rcFile}" APPEND PROPERTY COMPILE_FLAGS "${_allFlags}")
        else()
          set_property(SOURCE "${_rcFile}" PROPERTY COMPILE_FLAGS "${_allFlags}")
        endif()

        list(APPEND _targetRcFiles "${_rcFile}")
      endforeach()

      foreach(_importResId ${_importResIds})
        string(REPLACE ";" "\;" _importResId "${_importResId}") # [WA#1] Must escape ; again if occurred in item.

        set(_importedResources "")
        intel_rc_get_resource(_importedResources "${_importResId}")
        list(APPEND _targetRcFiles "${_importedResources}")
      endforeach()
    endforeach()
  endif()

  list(REMOVE_DUPLICATES _targetRcFiles)
  set("${rcFilesVarName}" "${_targetRcFiles}" PARENT_SCOPE)
endfunction()

# ========================================= Symbols management =========================================

# Excludes from symbol export specified targets.
#
# intel_target_exclude_from_symbol_export(
#     targetName
#     [nonExportableTarget [nonExportableTarget [...]]]
#   )
#
# @param targetName          CMake target where specified linked targets will be excluded
#                            from symbol export.
# @param nonExportableTarget Targets that will be excluded from symbol export (for target targetName).
function(intel_target_exclude_from_symbol_export targetName)
  if(NOT (TARGET "${targetName}"))
    message(FATAL_ERROR "The target \"${targetName}\" is not defined.")
  endif()

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    foreach(_configName ${CMAKE_CONFIGURATION_TYPES})
      string(REPLACE ";" "\;" _configName "${_configName}") # [WA#1] Must escape ; again if occurred in item.
      string(TOUPPER "${_configName}" _upperConfigName)

      set(_locPropName "LOCATION_${_upperConfigName}")

      set(_excludedExportTargets "")
      foreach(_excludedExportTarget ${ARGN})
        string(REPLACE ";" "\;" _excludedExportTarget "${_excludedExportTarget}") # [WA#1] Must escape ; again if occurred in item.

        if(TARGET "${_excludedExportTarget}")
          get_property(_locationSet TARGET "${_excludedExportTarget}" PROPERTY "${_locPropName}" SET)
          if(_locationSet)
            get_property(_location TARGET "${_excludedExportTarget}" PROPERTY "${_locPropName}")
          else()
            get_property(_location TARGET "${_excludedExportTarget}" PROPERTY LOCATION)
          endif()
        else()
          message(FATAL_ERROR "The excluded target \"${_excludedExportTarget}\" is not defined.")
        endif()

        if(NOT (_location MATCHES "(^|-NOTFOUND)$"))
          get_filename_component(_fileNameWoExt "${_location}" NAME_WE)

          if(NOT (_excludedExportTargets MATCHES "^$"))
            set(_excludedExportTargets "${_excludedExportTargets}:${_fileNameWoExt}")
          else()
            set(_excludedExportTargets "${_fileNameWoExt}")
          endif()
        endif()
      endforeach()

      set_property(TARGET "${targetName}" APPEND PROPERTY "LINK_FLAGS_${_upperConfigName}" "-Wl,--exclude-libs,${_excludedExportTargets}")
    endforeach()
  else()
    message(WARNING "Symbol exclusion works only for GCC-like compilers.")
  endif()
endfunction()

# ============================================ Source groups ===========================================

# Register source group (advanced version).
#
# intel_sg_register(
#     srcGrpId
#     filterLabel
#     [GROUPS [childSrcGrpId [childSrcGrpId [...]]]]
#     [FILES  [srcFile [srcFile [...]]]]
#     [REGULAR_EXPRESSION [srcRe [srcRe [...]]]]
#     [...]
#     [{GROUPS|FILES|REGULAR_EXPRESSION} ...]
#   )
#
# @param srcGrpId      Source group to register.
# @param filterLabel   Label of filter for source group. Last registered label is used.
# @param childSrcGrpId Child source group. Their filter label will be prepended with filter labels
#                      of ancestors source groups.
# @param srcFile       Source files registered in the group. (Last source group match will be used though.)
# @param srcRe         Regular expression that identify files added to source group.
function(intel_sg_register srcGrpId filterLabel)
  set(_sgPropPrefix "INTEL_CMAKE_PROPERTY__SRC_GROUPS_SG${srcGrpId}_")
  set(_sgFilterPropName "${_sgPropPrefix}FILTER")
  set(_sgGroupsPropName "${_sgPropPrefix}GROUPS")
  set(_sgFilesPropName  "${_sgPropPrefix}FILES")
  set(_sgRePropName     "${_sgPropPrefix}REGEX")

  set_property(GLOBAL PROPERTY "${_sgFilterPropName}" "${filterLabel}")
  get_property(_sgGroups GLOBAL PROPERTY "${_sgGroupsPropName}")
  get_property(_sgFiles  GLOBAL PROPERTY "${_sgFilesPropName}")
  get_property(_sgRe     GLOBAL PROPERTY "${_sgRePropName}")

  set(_parseState 2)
  foreach(_sgArg ${ARGN})
    string(REPLACE ";" "\;" _sgArg "${_sgArg}") # [WA#1] Must escape ; again if occurred in item.

    # States: [0] <param> [1] <param> [2] *( ( "GROUPS" | "FILES" | "REGULAR_EXPRESSION" ) [3] *<param> [3] ) )
    # Transitions: 0 -> 1 -> 2 // by explict parameters
    #              2 (GROUPS) -> 3
    #              2 (FILES) -> 3
    #              2 (REGULAR_EXPRESSION) -> 3
    #              3 (GROUPS) -> 3
    #              3 (FILES) -> 3
    #              3 (REGULAR_EXPRESSION) -> 3
    #              3 -> 3
    # Stop States: 2, 3
    if(_parseState EQUAL 2)
      if(_sgArg MATCHES "^GROUPS$")
        set(_sgCollection "_sgGroups")
        set(_parseState 3)
      elseif(_sgArg MATCHES "^FILES$")
        set(_sgCollection "_sgFiles")
        set(_parseState 3)
      elseif(_sgArg MATCHES "^REGULAR_EXPRESSION$")
        set(_sgCollection "_sgRe")
        set(_parseState 3)
      else()
        message(FATAL_ERROR "Invalid parameter token near \"${_sgArg}\".")
      endif()
    elseif(_parseState EQUAL 3)
      if(_sgArg MATCHES "^GROUPS$")
        set(_sgCollection "_sgGroups")
      elseif(_sgArg MATCHES "^FILES$")
        set(_sgCollection "_sgFiles")
      elseif(_sgArg MATCHES "^REGULAR_EXPRESSION$")
        set(_sgCollection "_sgRe")
      else()
        list(APPEND "${_sgCollection}" "${_sgArg}")
      endif()
    else()
      message(FATAL_ERROR "Invalid parameter token near \"${_sgArg}\".")
    endif()
  endforeach()
  if(NOT ((_parseState EQUAL 2) OR (_parseState EQUAL 3)))
    message(FATAL_ERROR "Invalid number of parameters.")
  endif()

  if(DEFINED _sgGroups)
    list(REMOVE_DUPLICATES _sgGroups)
  endif()
  if(DEFINED _sgFiles)
    list(REMOVE_DUPLICATES _sgFiles)
  endif()
  if(DEFINED _sgRe)
    list(REMOVE_DUPLICATES _sgRe)
  endif()

  set_property(GLOBAL PROPERTY "${_sgGroupsPropName}" "${_sgGroups}")
  set_property(GLOBAL PROPERTY "${_sgFilesPropName}"  "${_sgFiles}")
  set_property(GLOBAL PROPERTY "${_sgRePropName}"     "${_sgRe}")
endfunction()

# Defines filter in source group for specific source directory.
#
# @param srcGrpId Registered source group used to create filter definitions.
function(intel_sg_define srcGrpId)
  set(_sgPropPrefix "INTEL_CMAKE_PROPERTY__SRC_GROUPS_SG${srcGrpId}_")
  set(_sgFilterPropName "${_sgPropPrefix}FILTER")
  set(_sgGroupsPropName "${_sgPropPrefix}GROUPS")
  set(_sgFilesPropName  "${_sgPropPrefix}FILES")
  set(_sgRePropName     "${_sgPropPrefix}REGEX")

  get_property(_sgFilterSet GLOBAL PROPERTY "${_sgFilterPropName}" SET)
  if(NOT _sgFilterSet)
    return()
  endif()

  list(LENGTH ARGN _paramsCount)
  if(_paramsCount GREATER 0)
    list(GET ARGN 0 _sgFilterPrefix)
    string(REPLACE ";" "\;" _sgFilterPrefix "${_sgFilterPrefix}\\") # [WA#1] Must escape ; again if occurred in item.
  else()
    set(_sgFilterPrefix "")
  endif()

  get_property(_sgFilter GLOBAL PROPERTY "${_sgFilterPropName}")
  get_property(_sgGroups GLOBAL PROPERTY "${_sgGroupsPropName}")
  get_property(_sgFiles  GLOBAL PROPERTY "${_sgFilesPropName}")
  get_property(_sgRe     GLOBAL PROPERTY "${_sgRePropName}")

  foreach(_sgReItem ${_sgRe})
    string(REPLACE ";" "\;" _sgReItem "${_sgReItem}") # [WA#1] Must escape ; again if occurred in item.
    source_group("${_sgFilterPrefix}${_sgFilter}" REGULAR_EXPRESSION "${_sgReItem}")
  endforeach()
  source_group("${_sgFilterPrefix}${_sgFilter}" FILES ${_sgFiles})

  foreach(_sgGroup ${_sgGroups})
    string(REPLACE ";" "\;" _sgGroup "${_sgGroup}") # [WA#1] Must escape ; again if occurred in item.
    intel_sg_define("${_sgGroup}" "${_sgFilterPrefix}${_sgFilter}")
  endforeach()
endfunction()

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

#cmake_policy(POP) # Not needed... CMake manages policy scope.

