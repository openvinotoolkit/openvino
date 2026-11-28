if(ENABLE_TESTS AND NOT MEGAKERNEL_IMPLEMENTATION STREQUAL ""
        AND TARGET megakernel_test_lib AND TARGET openvino_intel_gpu_plugin)
    add_executable(ov_megakernel_integration_test "${CMAKE_CURRENT_LIST_DIR}/mk_ov_integration_test.cpp")
    target_link_libraries(ov_megakernel_integration_test PRIVATE openvino::runtime gtest_main megakernel_test_lib)
    add_dependencies(ov_megakernel_integration_test openvino_intel_gpu_plugin)
    add_test(NAME ov_megakernel_integration_test COMMAND ov_megakernel_integration_test)
endif()