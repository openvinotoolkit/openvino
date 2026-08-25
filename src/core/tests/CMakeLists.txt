# Standalone test driver for the Vulkan core (no openvino tree needed).
#
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#
# Vulkan discovery order:
#   1. an installed Vulkan SDK (find_package),
#   2. the vktools bootstrap in %LOCALAPPDATA%\vktools (see tools/setup_vktools.ps1).
#
# Every test_*.cpp in this folder becomes a <name>.exe linked against the core.

cmake_minimum_required(VERSION 3.20)
project(openvino_vulkan_core_tests CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ---- glslang -----------------------------------------------------------------
set(GLSLANG "" CACHE FILEPATH "Explicit path to glslang/glslangValidator")
if(GLSLANG)
    set(GLSLANG_VALIDATOR "${GLSLANG}")
else()
    find_program(GLSLANG_VALIDATOR NAMES glslang glslangValidator REQUIRED)
endif()

# ---- Vulkan ------------------------------------------------------------------
find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(VKTOOLS_DIR "$ENV{LOCALAPPDATA}/vktools")
    if(NOT EXISTS "${VKTOOLS_DIR}/vulkan/lib/vulkan-1.lib")
        message(FATAL_ERROR "No Vulkan SDK and no vktools bootstrap found.\n"
                            "Run: powershell -ExecutionPolicy Bypass -File tools/setup_vktools.ps1")
    endif()
    set(Vulkan_INCLUDE_DIR "${VKTOOLS_DIR}/vulkan/include")
    set(Vulkan_LIBRARY "${VKTOOLS_DIR}/vulkan/lib/vulkan-1.lib")
endif()

# ---- SPIR-V kernels ------------------------------------------------------------
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/spirv")
file(GLOB COMP_FILES "${CORE_DIR}/kernels/*.comp")
add_custom_command(
    OUTPUT "${GEN_DIR}/spirv_kernels.inc"
    COMMAND "${CMAKE_COMMAND}"
            -DGLSLANG="${GLSLANG_VALIDATOR}"
            -DSRC="${CORE_DIR}/kernels"
            -DOUT="${GEN_DIR}"
            -P "${CORE_DIR}/cmake/gen_spirv.cmake"
    DEPENDS ${COMP_FILES}
    COMMENT "Compiling native Vulkan kernels to SPIR-V")

# ---- core under test -----------------------------------------------------------
file(GLOB CORE_SRCS "${CORE_DIR}/*.cpp")
file(GLOB RUNTIME_SRCS "${CORE_DIR}/runtime/*.cpp")
add_library(vk_core_under_test STATIC
            ${CORE_SRCS}
            ${RUNTIME_SRCS}
            "${GEN_DIR}/spirv_kernels.inc")
target_include_directories(vk_core_under_test
    PUBLIC "${CORE_DIR}" "${Vulkan_INCLUDE_DIR}"
    PRIVATE "${GEN_DIR}")
target_link_libraries(vk_core_under_test PRIVATE "${Vulkan_LIBRARY}")

# ---- tests ---------------------------------------------------------------------
file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name "${test_src}" NAME_WE)
    string(REGEX REPLACE "^test_" "" exe_name "${test_name}")
    add_executable(${exe_name} "${test_src}")
    target_link_libraries(${exe_name} PRIVATE vk_core_under_test)
endforeach()

enable_testing()
foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name "${test_src}" NAME_WE)
    string(REGEX REPLACE "^test_" "" exe_name "${test_name}")
    add_test(NAME "${exe_name}" COMMAND "${exe_name}")
endforeach()
