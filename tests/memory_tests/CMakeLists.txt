function(add_test)
    set(EXE_NAME "test_${ARGV0}")
    add_executable(${EXE_NAME} "src/${EXE_NAME}.cpp")
    target_link_libraries(${EXE_NAME} openvino::runtime)
endfunction()

add_test(inference_sync)
add_test(inference_async)
