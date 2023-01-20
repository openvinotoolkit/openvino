
ov_add_frontend(NAME tensorflow_lite
        LINKABLE_FRONTEND
        FILEDESCRIPTION "FrontEnd to load and convert TensorFlow Lite file format"
        LINK_LIBRARIES openvino::util openvino::runtime::dev)

set(TARGET_NAME "${FRONTEND_NAME_PREFIX}tensorflow_lite${FRONTEND_NAME_SUFFIX}")
target_link_libraries(${TARGET_NAME} PRIVATE openvino::frontend::tensorflow_common)