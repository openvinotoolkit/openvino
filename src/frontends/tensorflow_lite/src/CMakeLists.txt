
ov_add_frontend(NAME tensorflow_lite
        LINKABLE_FRONTEND
        FILEDESCRIPTION "FrontEnd to load and convert TensorFlow Lite file format"
        LINK_LIBRARIES openvino::util openvino::runtime::dev
        openvino_tensorflow_frontend_common_logic)

set(frontend_target_name "${FRONTEND_NAME_PREFIX}tensorflow_lite${FRONTEND_NAME_SUFFIX}")
target_link_libraries(${frontend_target_name} PUBLIC openvino::frontend::tensorflow)