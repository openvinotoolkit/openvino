# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# installation directory
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Intel")

# TODO: provide icons
# set(CPACK_NSIS_MUI_ICON "")
# set(CPACK_NSIS_MUI_UNIICON "${CPACK_NSIS_MUI_ICON}")
# set(CPACK_NSIS_MUI_WELCOMEFINISHPAGE_BITMAP "")
# set(CPACK_NSIS_MUI_UNWELCOMEFINISHPAGE_BITMAP "")
# set(CPACK_NSIS_MUI_HEADERIMAGE "")

# we allow to install several packages at once
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL OFF)
set(CPACK_NSIS_MODIFY_PATH OFF)

set(CPACK_NSIS_DISPLAY_NAME "Intel(R) OpenVINO(TM) ${OpenVINO_VERSION}")
set(CPACK_NSIS_PACKAGE_NAME "Intel(R) OpenVINO(TM) ToolKit, v. ${OpenVINO_VERSION}.${OpenVINO_PATCH_VERSION}")

# contact
set(CPACK_NSIS_CONTACT "CPACK_NSIS_CONTACT")

# links in menu
set(CPACK_NSIS_MENU_LINKS
		"https://docs.openvinoo.ai" "OpenVINO Documentation")

# welcome and finish titles
set(CPACK_NSIS_WELCOME_TITLE "Welcome to Intel(R) Distribution of OpenVINO(TM) Toolkit installation")
set(CPACK_NSIS_FINISH_TITLE "")

# autoresize?
set(CPACK_NSIS_MANIFEST_DPI_AWARE ON)

# branding text
set(CPACK_NSIS_BRANDING_TEXT "Intel(R) Corp.")
set(CPACK_NSIS_BRANDING_TEXT_TRIM_POSITION RIGHT)

# don't set this variable since we need a user to agree with a lincense
# set(CPACK_NSIS_IGNORE_LICENSE_PAGE OFF)
