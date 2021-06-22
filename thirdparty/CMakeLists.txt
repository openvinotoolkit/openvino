# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ittapi)
add_subdirectory(itt_collector)
add_subdirectory(xbyak EXCLUDE_FROM_ALL)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)

#
# Pugixml
#

function(ie_build_pugixml)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(pugixml EXCLUDE_FROM_ALL)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml)
endfunction()

if(ENABLE_SYSTEM_PUGIXML)
    find_package(PugiXML REQUIRED)
    set_property(TARGET pugixml PROPERTY IMPORTED_GLOBAL TRUE)
else()
    ie_build_pugixml()
endif()

#
# Export
#

openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)
