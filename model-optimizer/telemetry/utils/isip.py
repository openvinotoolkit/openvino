# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from platform import system


class ISIPConsent(Enum):
    APPROVED = 0
    DECLINED = 1
    UNKNOWN = 2


def isip_consent_base_dir():
    """
    Returns the base directory with the ISIP consent file. The full directory may not have write access on Linux/OSX
    systems so that is why the base directory is used.
    :return:
    """
    platform = system()

    dir_to_check = None

    if platform == 'Windows':
        dir_to_check = '$LOCALAPPDATA'
    elif platform in ['Linux', 'Darwin']:
        dir_to_check = '$HOME'

    if dir_to_check is None:
        raise Exception('Failed to find location of the ISIP consent')

    return os.path.expandvars(dir_to_check)


def _isip_consent_sub_directory():
    platform = system()
    if platform == 'Windows':
        return 'Intel Corporation'
    elif platform in ['Linux', 'Darwin']:
        return 'intel'
    raise Exception('Failed to find location of the ISIP consent')


def _isip_consent_dir():
    dir_to_check = os.path.join(isip_consent_base_dir(), _isip_consent_sub_directory())
    return os.path.expandvars(dir_to_check)


def _isip_consent_file():
    return os.path.join(_isip_consent_dir(), 'isip')


def isip_consent():
    file_to_check = _isip_consent_file()
    if not os.path.exists(file_to_check):
        return ISIPConsent.UNKNOWN

    try:
        with open(file_to_check, 'r') as file:
            content = file.readline().strip()
            if content == '1':
                return ISIPConsent.APPROVED
            else:
                return ISIPConsent.DECLINED
    except Exception as e:
        pass

    # unknown value in the file is considered as a unknown consent
    return ISIPConsent.UNKNOWN
