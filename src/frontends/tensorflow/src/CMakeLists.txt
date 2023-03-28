# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

list(APPEND CUSTOM_LINK_LIBRARIES
	openvino::core::dev
	openvino::frontend::tensorflow_common
)

if(ENABLE_SNAPPY_COMPRESSION)
	list(APPEND CUSTOM_LINK_LIBRARIES
		snappy
	)
endif()

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES ${CUSTOM_LINK_LIBRARIES})

if(ENABLE_SNAPPY_COMPRESSION)
	target_compile_definitions(openvino_tensorflow_frontend PUBLIC ENABLE_SNAPPY_COMPRESSION)
endif()
