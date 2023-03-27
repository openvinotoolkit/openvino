# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


add_subdirectory(src)

# TODO Reenable PDPD after paddlepaddle==2.5.0 with compliant protobuf is released (ticket 95904)
#if(ENABLE_TESTS)
#    add_subdirectory(tests)
#endif()
