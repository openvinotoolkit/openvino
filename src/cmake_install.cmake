# Install script for directory: D:/OPENVINO/openvino/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/OpenVINO")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tbb" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/3rdparty/tbb" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/temp/tbb/bin")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tbb" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/3rdparty/tbb" TYPE FILE RENAME "TBB-LICENSE" FILES "D:/OPENVINO/openvino/temp/tbb/LICENSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tbb_dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/3rdparty/tbb/lib" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/temp/tbb/lib/cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tbb_dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/3rdparty/tbb" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/temp/tbb/include")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tbb_dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/3rdparty/tbb" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/temp/tbb/lib" REGEX "/cmake$" EXCLUDE)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/common/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/core/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/frontends/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/plugins/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/inference/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/Debug" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvinod.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/Release" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/MinSizeRel" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/RelWithDebInfo" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Debug" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvinod.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/MinSizeRel" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/RelWithDebInfo" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/include/openvino_runtime_dev" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/src/inference/dev_api/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_dev" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/runtime/cmake/OpenVINOTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/runtime/cmake/OpenVINOTargets.cmake"
         "D:/OPENVINO/openvino/src/CMakeFiles/Export/5ecc6fb595e257054a4227a562196a09/OpenVINOTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/runtime/cmake/OpenVINOTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/runtime/cmake/OpenVINOTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/src/CMakeFiles/Export/5ecc6fb595e257054a4227a562196a09/OpenVINOTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/src/CMakeFiles/Export/5ecc6fb595e257054a4227a562196a09/OpenVINOTargets-debug.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/src/CMakeFiles/Export/5ecc6fb595e257054a4227a562196a09/OpenVINOTargets-minsizerel.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/src/CMakeFiles/Export/5ecc6fb595e257054a4227a562196a09/OpenVINOTargets-relwithdebinfo.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/src/CMakeFiles/Export/5ecc6fb595e257054a4227a562196a09/OpenVINOTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/cmake" TYPE FILE FILES
    "D:/OPENVINO/openvino/share/OpenVINOConfig.cmake"
    "D:/OPENVINO/openvino/OpenVINOConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/bindings/cmake_install.cmake")
endif()

