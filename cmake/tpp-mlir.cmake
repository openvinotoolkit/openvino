# If TPP-MLIR is in library path, add it to the dependencies
# This should be the build directory, not the source or the 'lib'
# FIXME: Make this an actual CMake discovery
if (TPP_MLIR_DIR)
    message(STATUS "TPP-MLIR at ${TPP_MLIR_DIR}")
    add_compile_definitions(TPP_MLIR)
    set(TPP_MLIR_LIBS
            # Keep the next two libs at the top of the list to avoid undefined references at link time
            TPPPipeline
            TPPPassBundles
            TPPCheckDialect
            TPPCheckToLoops
            TPPGPU
            TPPIR
            TPPLinalgToFunc
            TPPLinalgToXSMM
            TPPPerfDialect
            TPPPerfToFunc
            TPPPerfToLoop
            TPPRunner
            TPPTestLib
            TPPTransforms
            TPPTransformsUtils
            TPPXsmmDialect
            TPPXsmmToFunc
            xsmm
            tpp_xsmm_runner_utils
        )
    function(add_tpp_mlir_includes target)
        target_include_directories(${target} PRIVATE ${TPP_MLIR_DIR}/../include ${TPP_MLIR_DIR}/include)
    endfunction()
    function(add_tpp_mlir_libs target)
        target_link_directories(${target} PRIVATE ${TPP_MLIR_DIR}/lib)
        target_link_libraries(${target} PRIVATE ${TPP_MLIR_LIBS})
        target_link_options(${target} PRIVATE
            -Wl,--no-as-needed
            -L${TPP_MLIR_DIR}/lib
            -ltpp_xsmm_runner_utils
            -L${LLVM_LIBRARY_DIR}
            -lmlir_c_runner_utils
            -Wl,--as-needed
        )
        #FIXME: Provide platform-independent way of doing that:
        install(FILES ${TPP_MLIR_DIR}/lib/libtpp_xsmm_runner_utils.so ${TPP_MLIR_DIR}/lib/libtpp_xsmm_runner_utils.so.19.0git DESTINATION ${OV_CPACK_RUNTIMEDIR})
    endfunction()
else()
    function(add_tpp_mlir_includes target)
        message(DEBUG "TPP-MLIR not enabled, skipping ${target}")
    endfunction()
    function(add_tpp_mlir_libs target)
        message(DEBUG "TPP-MLIR not enabled, skipping ${target}")
    endfunction()
endif()
