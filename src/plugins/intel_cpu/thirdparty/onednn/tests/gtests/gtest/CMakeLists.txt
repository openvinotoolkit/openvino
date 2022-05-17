# Do not modify this file

# XXX: DPCPP issues a warning when WINAPI is used:
# warning: '__stdcall' calling convention is not supported for this target
if(WIN32 AND DNNL_WITH_SYCL)
    append(CMAKE_CXX_FLAGS "-Wno-ignored-attributes")
endif()

set(TARGET_NAME dnnl_gtest)
set(MAIN_SRC "src/gtest-all.cc")
source_group("" FILES ${MAIN_SRC})
include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR}
                                       ${CMAKE_CURRENT_SOURCE_DIR}/..
)

find_package(Threads REQUIRED)
add_library(${TARGET_NAME} STATIC ${MAIN_SRC})

# XXX: Link pthread explicitly: C and C++ compiler may have different flags.
# In that case FindThreads module may skip adding any flags for pthread library
# because they are not needed for C compiler but this may led to issues with
# C++ compiler which is not checked.
if(UNIX OR MINGW)
    target_link_libraries(${TARGET_NAME} pthread)
endif()
