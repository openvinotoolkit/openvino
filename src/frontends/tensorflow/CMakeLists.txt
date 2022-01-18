# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO: Add LINKABLE_FRONTEND option when tensorflow frontend directory is moved to openvino folder
ov_add_frontend(NAME tensorflow
                SHUTDOWN_PROTOBUF
                SKIP_INSTALL
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util)

# TODO: once TensorFlow FE become releasable component, need to remove these lines and SKIP_INSTALL above
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
