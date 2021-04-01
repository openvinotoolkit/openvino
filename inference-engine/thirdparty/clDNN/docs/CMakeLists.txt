# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "docs")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}")

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__main         "configurations")
set(__CLDNN_CfgFile__main
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
  )

set(__CLDNN_Directory__examples "${CMAKE_CURRENT_SOURCE_DIR}/../tests/examples")
set(__CLDNN_Label__examples     "examples")
file(GLOB __CLDNN_Sources__examples
    "${__CLDNN_Directory__examples}/*.h"
    "${__CLDNN_Directory__examples}/*.hpp"
    "${__CLDNN_Directory__examples}/*.cpp"
  )

set(__CLDNN_Directory__images "${CMAKE_CURRENT_SOURCE_DIR}/img")
set(__CLDNN_Label__images     "images")
file(GLOB __CLDNN_Sources__images
    "${__CLDNN_Directory__images}/*.jpg"
    "${__CLDNN_Directory__images}/*.png"
    "${__CLDNN_Directory__images}/*.gif"
    "${__CLDNN_Directory__images}/*.svg"
  )


set(__CLDNN_AllFiles
    ${__CLDNN_CfgFile__main}
    ${__CLDNN_Sources__examples}
    ${__CLDNN_Sources__images}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"     FILES ${__CLDNN_CfgFile__main})
source_group("${__CLDNN_Label__examples}" FILES ${__CLDNN_Sources__examples})
source_group("${__CLDNN_Label__images}"   FILES ${__CLDNN_Sources__images})

# ========================= Custom steps of custom target (always invalid) =============================

# Check for dot executable presense.
find_program(CLDNN_FIND__GRAPHVIZ_DOT_BINPATH
    NAMES dot
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_FIND_ROOT_PATH
  )
if(CLDNN_FIND__GRAPHVIZ_DOT_BINPATH)
  set(__CLDNN_HaveDot "YES")
else()
  set(__CLDNN_HaveDot "NO")
endif()

# Use correct "cat".
if(CMAKE_HOST_WIN32)
  set(__CLDNN_CatBinName "type") # type is special part of cmd and cannot be detected by find_program().
else()
  set(__CLDNN_CatBinName "cat")
endif()

# Generating documentation from code based on configuration file.
add_custom_target("${CLDNN_BUILD__PROJ}"
    "${CMAKE_COMMAND}" -E make_directory "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/"
    COMMAND "${CMAKE_COMMAND}" -E echo "" && "(" ${__CLDNN_CatBinName} \"${__CLDNN_CfgFile__main}\" && "${CMAKE_COMMAND}" -E echo \"OUTPUT_DIRECTORY = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/\" && "${CMAKE_COMMAND}" -E echo \"HAVE_DOT = ${__CLDNN_HaveDot}\" ")" "|"  doxygen -
    DEPENDS ${__CLDNN_AllFiles}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation using Doxygen..."
    SOURCES ${__CLDNN_AllFiles}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

unset(CLDNN_FIND__GRAPHVIZ_DOT_BINPATH CACHE)
unset(__CLDNN_HaveDot)
unset(__CLDNN_CatBinName)

# ======================================================================================================
