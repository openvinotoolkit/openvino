# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                SKIP_INSTALL
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util)

# TODO: once TensorFlow FE become releasable component, need to remove these lines and SKIP_INSTALL above
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
