# If the pyx file is a C++ file, we should specify that here.
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (TARGET_NAME "ie_api")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set_source_files_properties(
    ie_api_impl_defs.pxd
    ie_api_impl.hpp
    ie_api_impl.cpp
    ie_api.pyx
    ie_api.pxd

  PROPERTIES CYTHON_IS_CXX TRUE
)

cython_add_module (
    ${TARGET_NAME}

    ie_api_impl_defs.pxd
    ie_api_impl.hpp
    ie_api_impl.cpp
    ie_api.pyx
)

set_target_properties (${TARGET_NAME} PROPERTIES CXX_STANDARD 11 LINKER_LANGUAGE CXX)
target_link_libraries (${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# perform copy
ADD_CUSTOM_COMMAND (TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/inference_engine/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/requirements.txt ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../../requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../__init__.py
)