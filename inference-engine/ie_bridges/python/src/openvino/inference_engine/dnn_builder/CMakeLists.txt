# If the pyx file is a C++ file, we should specify that here.
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(TARGET_NAME "dnn_builder")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine/${TARGET_NAME})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set_source_files_properties(
    dnn_builder_defs.pxd
    dnn_builder_impl.hpp
    dnn_builder_impl.cpp
    dnn_builder.pyx
    dnn_builder.pxd

  PROPERTIES CYTHON_IS_CXX TRUE
)

cython_add_module(
    ${TARGET_NAME}

    dnn_builder_impl_defs.pxd
    dnn_builder_impl.hpp
    dnn_builder_impl.cpp
    dnn_builder.pyx
)

set_target_properties (${TARGET_NAME} PROPERTIES CXX_STANDARD 11 LINKER_LANGUAGE CXX)
add_dependencies (${TARGET_NAME} ie_api)
target_include_directories (${TARGET_NAME} PRIVATE ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/inference_engine )
target_link_libraries (${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# perform copy
ADD_CUSTOM_COMMAND (TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/inference_engine/${TARGET_NAME}/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)