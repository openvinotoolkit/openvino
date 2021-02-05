project (ng_bench)
# Setup Conan package manager
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake" 
                 TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

# Install all dependent libraries with Conan
conan_cmake_run(CONANFILE conanfile.txt
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

# 
add_executable(ng_bench one_hot.cpp)
target_link_libraries(ng_bench PRIVATE CONAN_PKG::benchmark
                                        ngraph_test_util
                                        ngraph
                                        ngraph::builder
                                        ngraph_backend)
