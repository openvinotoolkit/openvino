if(EXISTS "$ENV{MV_COMMON_BASE}")
    set(MV_COMMON_BASE $ENV{MV_COMMON_BASE})
else()
    set(MV_COMMON_BASE ${CMAKE_CURRENT_LIST_DIR}/..)
endif(EXISTS "$ENV{MV_COMMON_BASE}")

if(NOT WIN32)
    find_package(Threads REQUIRED)

    find_path(LIBUSB_INCLUDE_DIR NAMES libusb.h PATH_SUFFIXES "include" "libusb" "libusb-1.0")
    find_library(LIBUSB_LIBRARY NAMES usb-1.0 PATH_SUFFIXES "lib")

    if(NOT LIBUSB_INCLUDE_DIR OR NOT LIBUSB_LIBRARY)
        message(FATAL_ERROR "libusb is required")
    endif()
endif()

set(XLINK_INCLUDE
        ${MV_COMMON_BASE}/XLink/pc
        ${MV_COMMON_BASE}/XLink/shared
        ${MV_COMMON_BASE}/shared/include
        )

set(XLINK_INCLUDE_DIRECTORIES
        ${XLINK_INCLUDE}
        ${LIBUSB_INCLUDE_DIR}
        )

set(XLINK_SOURCES
        ${MV_COMMON_BASE}/XLink/pc/XLinkPlatform.c
        ${MV_COMMON_BASE}/XLink/pc/usb_boot.c
        ${MV_COMMON_BASE}/XLink/pc/pcie_host.c
        ${MV_COMMON_BASE}/XLink/shared/XLinkDeprecated.c
        ${MV_COMMON_BASE}/XLink/shared/XLinkPrivateFields.c
        ${MV_COMMON_BASE}/XLink/shared/XLinkDispatcherImpl.c
        ${MV_COMMON_BASE}/XLink/shared/XLinkDevice.c
        ${MV_COMMON_BASE}/XLink/shared/XLinkStream.c
        ${MV_COMMON_BASE}/XLink/shared/XLinkDispatcher.c
        ${MV_COMMON_BASE}/shared/src/mvStringUtils.c
        )
