find_package(OpenCL REQUIRED)

add_library(TestMegaKernel SHARED megakernelImpl.cpp impl/test_runtime.cpp)
include(GenerateExportHeader)
generate_export_header(TestMegaKernel EXPORT_FILE_NAME exportApi.h EXPORT_MACRO_NAME TEST_MEGAKERNEL_EXPORT)
target_include_directories(TestMegaKernel PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_SOURCE_DIR}/src/plugins/intel_gpu/src/megakernel")
target_link_libraries(TestMegaKernel PRIVATE OpenCL::OpenCL)

if(ENABLE_TESTS)
    add_executable(ov_megakernel_integration_test "${CMAKE_CURRENT_LIST_DIR}/../megakernel_test.cpp")
    target_link_libraries(ov_megakernel_integration_test PRIVATE openvino::runtime gtest_main TestMegaKernel)
    add_dependencies(ov_megakernel_integration_test openvino_intel_gpu_plugin${OV_GPU_SUFFIX})
    add_test(NAME ov_megakernel_integration_test COMMAND ov_megakernel_integration_test)
endif()