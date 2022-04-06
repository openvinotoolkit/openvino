# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    # needed by install_prerequisites which call extract_release_version as python script
    from version import extract_release_version, get_version
except ImportError:
    from openvino.tools.mo.utils.version import extract_release_version, get_version


if __name__ == "__main__":
    print("{}.{}".format(*extract_release_version(get_version())))
