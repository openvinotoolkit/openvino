# Install script for directory: D:/OPENVINO/openvino

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

# Set default install directory permissions.
if(NOT DEFINED CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS)
  set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS "OWNER_READ;OWNER_WRITE;OWNER_EXECUTE;GROUP_READ;GROUP_EXECUTE;OWNER_EXECUTE;WORLD_READ;WORLD_EXECUTE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/thirdparty/ocl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/thirdparty/onnx/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/include/nlohmann_json" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/thirdparty/json/nlohmann_json/include")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cpp_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/cpp/thirdparty" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/thirdparty/gflags" REGEX "/bazel$" EXCLUDE REGEX "/doc$" EXCLUDE REGEX "/\\.git$" EXCLUDE REGEX "/appveyor\\.yml$" EXCLUDE REGEX "/authors\\.txt$" EXCLUDE REGEX "/build$" EXCLUDE REGEX "/changelog\\.txt$" EXCLUDE REGEX "/\\.gitattributes$" EXCLUDE REGEX "/\\.gitignore$" EXCLUDE REGEX "/\\.gitmodules$" EXCLUDE REGEX "/test$" EXCLUDE REGEX "/install\\.md$" EXCLUDE REGEX "/readme\\.md$" EXCLUDE REGEX "/\\.travis\\.yml$" EXCLUDE REGEX "/src\\/gflags\\_completions\\.sh$" EXCLUDE REGEX "/workspace$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cpp_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/cpp/thirdparty/zlib/zlib" TYPE FILE FILES
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/adler32.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/compress.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/crc32.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/crc32.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/deflate.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/deflate.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/gzclose.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/gzguts.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/gzlib.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/gzread.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/gzwrite.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/infback.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inffast.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inffast.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inffixed.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inflate.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inflate.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inftrees.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/inftrees.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/trees.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/trees.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/uncompr.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/zconf.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/zlib.h"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/zutil.c"
    "D:/OPENVINO/openvino/thirdparty/zlib/zlib/zutil.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cpp_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/cpp/thirdparty/zlib" TYPE FILE FILES "D:/OPENVINO/openvino/thirdparty/zlib/CMakeLists.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cpp_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/cpp/thirdparty" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/thirdparty/json/nlohmann_json" REGEX "/changelog\\.md$" EXCLUDE REGEX "/citation\\.cff$" EXCLUDE REGEX "/\\.clang\\-format$" EXCLUDE REGEX "/\\.clang\\-tidy$" EXCLUDE REGEX "/docs$" EXCLUDE REGEX "/\\.git$" EXCLUDE REGEX "/\\.github$" EXCLUDE REGEX "/\\.gitignore$" EXCLUDE REGEX "/\\.lgtm\\.yml$" EXCLUDE REGEX "/makefile$" EXCLUDE REGEX "/meson\\.build$" EXCLUDE REGEX "/readme\\.md$" EXCLUDE REGEX "/\\.reuse$" EXCLUDE REGEX "/tests$" EXCLUDE REGEX "/tools$" EXCLUDE REGEX "/wsjcpp\\.yml$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cpp_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/cpp/thirdparty" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/thirdparty/cnpy")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/samples/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/plugins/template/backend/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES
    "D:/OPENVINO/openvino/share/OpenVINODeveloperPackageConfig.cmake"
    "D:/OPENVINO/openvino/share/OpenVINODeveloperPackageConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/cmake/developer_package/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/CMakeCache.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/pugixmld.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/pugixml.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/pugixml.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/pugixml.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/gflags_nothreads_static_debug.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/gflags_nothreads_static.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/gflags_nothreads_static.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/gflags_nothreads_static.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_ittd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_itt.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_itt.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_itt.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_utild.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_util.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_util.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_util.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_snippetsd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_snippets.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_snippets.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_snippets.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_offline_transformationsd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_offline_transformations.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_offline_transformations.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_offline_transformations.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_referenced.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_reference.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_reference.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_reference.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_shape_inferenced.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_shape_inference.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_shape_inference.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_shape_inference.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_runtime_sd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_runtime_s.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_runtime_s.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_runtime_s.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/format_readerd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/format_reader.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/format_reader.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/format_reader.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/ie_samples_utilsd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/ie_samples_utils.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/ie_samples_utils.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/ie_samples_utils.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Debug/openvino_interpreter_backendd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/Release/openvino_interpreter_backend.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/openvino_interpreter_backend.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/lib" TYPE STATIC_LIBRARY FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/openvino_interpreter_backend.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "developer_package")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/developer_package/cmake/OpenVINODeveloperPackageTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/developer_package/cmake/OpenVINODeveloperPackageTargets.cmake"
         "D:/OPENVINO/openvino/CMakeFiles/Export/c2899e1b3514df519fd6944cfe79fdea/OpenVINODeveloperPackageTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/developer_package/cmake/OpenVINODeveloperPackageTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/developer_package/cmake/OpenVINODeveloperPackageTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/CMakeFiles/Export/c2899e1b3514df519fd6944cfe79fdea/OpenVINODeveloperPackageTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/CMakeFiles/Export/c2899e1b3514df519fd6944cfe79fdea/OpenVINODeveloperPackageTargets-debug.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/CMakeFiles/Export/c2899e1b3514df519fd6944cfe79fdea/OpenVINODeveloperPackageTargets-minsizerel.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/CMakeFiles/Export/c2899e1b3514df519fd6944cfe79fdea/OpenVINODeveloperPackageTargets-relwithdebinfo.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/developer_package/cmake" TYPE FILE FILES "D:/OPENVINO/openvino/CMakeFiles/Export/c2899e1b3514df519fd6944cfe79fdea/OpenVINODeveloperPackageTargets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/build-modules/template/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/build-modules/template_extension/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/docs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/tools/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/scripts/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/licensing/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/OPENVINO/openvino/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
