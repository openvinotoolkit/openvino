# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_ie_version():
    try:
        from openvino.runtime import get_version  # pylint: disable=import-error,no-name-in-module
        return get_version()
    except:
        return None


if __name__ == "__main__":
    print(get_ie_version())
