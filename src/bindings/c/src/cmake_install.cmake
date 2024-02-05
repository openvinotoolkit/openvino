# Install script for directory: D:/OPENVINO/openvino/src/bindings/c/src

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_c" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/Debug" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_cd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/Release" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_c.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/MinSizeRel" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_c.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/RelWithDebInfo" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_c.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_c" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Debug" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_cd.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_c.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/MinSizeRel" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_c.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/RelWithDebInfo" TYPE SHARED_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_c.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_c_dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/include/openvino" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/src/bindings/c/include/openvino/")
endif()

