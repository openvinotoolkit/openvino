# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY samples/
        DESTINATION samples/python
        USE_SOURCE_PERMISSIONS
        COMPONENT python_samples)

ie_cpack(${PYTHON_COMPONENT} python_samples)
