find_package(OpenCL REQUIRED)

add_library(megakernel_test_lib SHARED mk_lib_api.cpp src/runtime.cpp)
include(GenerateExportHeader)
generate_export_header(megakernel_test_lib EXPORT_FILE_NAME exportApi.h EXPORT_MACRO_NAME TEST_MEGAKERNEL_EXPORT)
target_include_directories(megakernel_test_lib PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_SOURCE_DIR}/src/plugins/intel_gpu/src/megakernel_interface"
    "${CMAKE_SOURCE_DIR}/src/plugins/intel_gpu/src/megakernel")
target_compile_definitions(megakernel_test_lib PUBLIC CL_TARGET_OPENCL_VERSION=300)
target_link_libraries(megakernel_test_lib PRIVATE OpenCL::OpenCL)