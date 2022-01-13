set(TARGET_NAME ov_protobuf_shutdown)
add_library(${TARGET_NAME} STATIC shutdown_protobuf.cpp)
target_sources(${TARGET_NAME} INTERFACE shutdown_protobuf.cpp)
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>)
