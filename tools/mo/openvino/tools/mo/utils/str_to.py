# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class StrTo(object):
    @staticmethod
    def tuple(type_of_elements: type, string: str):
        if type_of_elements == int:
            string = string.replace('L', '')
        return tuple(type_of_elements(x) for x in string[1:-1].split(',') if x != '')

    @staticmethod
    def list(string: str, type_of_elements: type, sep: str):
        result = string.split(sep)
        result = [type_of_elements(x) for x in result]
        return result

    @staticmethod
    def bool(val: str):
        if val.lower() == "false":
            return False
        elif val.lower() == "true":
            return True
        else:
            raise ValueError("Value is not boolean: " + val)
