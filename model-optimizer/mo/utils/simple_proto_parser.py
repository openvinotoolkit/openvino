"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import ast
import logging as log
import os

from mo.utils.error import Error


class SimpleProtoParser(object):
    """
    This is a simple Proto2 parser that has limited functionality and is intended to parse configuration files for the
    models created with Object Detection API only. The result of the parser is the dictionary.
    """

    _tokens = list()
    _result = dict()

    def __init__(self):
        self._tokens = list()
        self._result = dict()

    @staticmethod
    def _convert_value_to_correct_datatype(value: str):
        """
        Converts string representation of the token to a value with proper data type.
        :param value: string representation to be converted.
        :return: converted to a correct data type value.
        """
        if value == 'true':
            return True
        if value == 'false':
            return False
        try:
            result = ast.literal_eval(value)
            return result
        except Exception:  # if it is not possible to evaluate the value then consider it as a string
            return value

    @staticmethod
    def _convert_values_to_correct_datatypes(d: dict):
        """
        Convert dictionary with values to correct data types.
        :param d: dictionary with values.
        :return: None
        """
        for key, value in d.items():
            if isinstance(value, dict):
                __class__._convert_values_to_correct_datatypes(value)
            elif isinstance(value, list):
                d[key] = [__class__._convert_value_to_correct_datatype(item) for item in value]
            else:
                d[key] = __class__._convert_value_to_correct_datatype(value)

    def _add_non_empty_token(self, token: str):
        """
        Add token to the list of tokens if it is non-empty.
        :param token: token to add
        :return: None
        """
        if token != "":
            self._tokens.append(token)

    def _parse_list(self, result: list, token_ind: int):
        prev_token = '['
        while token_ind < len(self._tokens):
            cur_token = self._tokens[token_ind]
            if cur_token == ']':
                return token_ind + 1
            if cur_token == ',':
                if prev_token == ',' or prev_token == '[':
                    raise Error('Missing value in the list at position {}'.format(token_ind))
            else:
                result.append(cur_token)
            token_ind += 1
            prev_token = cur_token
        return token_ind

    def _parse_tokens(self, result: dict, token_ind: int, depth: int=0):
        """
        Internal function that parses tokens.
        :param result: current dictionary where to store parse result.
        :param token_ind: index of the token from the tokens list to start parsing from.
        :return: token index to continue parsing from.
        """
        while token_ind < len(self._tokens):
            cur_token = self._tokens[token_ind]
            if cur_token == ',':  # redundant commas that we simply ignore everywhere except list "[x, y, z...]"
                token_ind += 1
                continue
            if cur_token == '}':
                return token_ind + 1
            next_token = self._tokens[token_ind + 1]
            if next_token == '{':
                result[cur_token] = dict()
                token_ind = self._parse_tokens(result[cur_token], token_ind + 2, depth + 1)
            elif next_token == ':':
                next_next_token = self._tokens[token_ind + 2]
                if next_next_token == '[':
                    result[cur_token] = list()
                    token_ind = self._parse_list(result[cur_token], token_ind + 3)
                else:
                    if cur_token not in result:
                        result[cur_token] = self._tokens[token_ind + 2]
                    else:
                        if not isinstance(result[cur_token], list):
                            old_val = result[cur_token]
                            result[cur_token] = [old_val]
                        result[cur_token].append(self._tokens[token_ind + 2])
                    token_ind += 3
            else:
                raise Error('Wrong character "{}" in position {}'.format(next_token, token_ind))
        if depth != 0:
            raise Error('Input/output braces mismatch.')
        return token_ind

    def _convert_tokens_to_dict(self):
        """
        Convert list of tokens into a dictionary with proper structure.
        Then converts values in the dictionary to values of correct data types. For example, 'false' -> False,
        'true' -> true, '0.004' -> 0.004, etc.
        :return: True if conversion is successful.
        """
        try:
            self._parse_tokens(self._result, 0)
        except Exception as ex:
            log.error('Failed to convert tokens to dictionary: {}'.format(str(ex)))
            return False
        self._convert_values_to_correct_datatypes(self._result)
        return True

    def _split_to_tokens(self, file_content: str):
        """
        The function gets file content as string and converts it to the list of tokens (all tokens are still strings).
        :param file_content: file content as a string
        """
        cur_token = ''
        string_started = False
        for line in file_content.split('\n'):
            cur_token = ''
            line = line.strip()
            if line.startswith('#'):  # skip comments
                continue
            for char in line:
                if string_started:
                    if char == '"':  # string ended
                        self._add_non_empty_token(cur_token)
                        cur_token = ''  # start of a new string
                        string_started = False
                    else:
                        cur_token += char
                elif char == '"':
                    self._add_non_empty_token(cur_token)
                    cur_token = ''  # start of a new string
                    string_started = True
                elif (char == " " and not string_started) or char == '\n':
                    self._add_non_empty_token(cur_token)
                    cur_token = ''
                elif char in [':', '{', '}', '[', ']', ',']:
                    self._add_non_empty_token(cur_token)
                    self._tokens.append(char)
                    cur_token = ''
                else:
                    cur_token += char
            self._add_non_empty_token(cur_token)
        self._add_non_empty_token(cur_token)

    def parse_from_string(self, file_content: str):
        """
        Parses the proto text file passed as a string.
        :param file_content: content of the file.
        :return: dictionary with file content or None if the file cannot be parsed.
        """
        self._split_to_tokens(file_content)
        if not self._convert_tokens_to_dict():
            log.error('Failed to generate dictionary representation of file.')
            return None
        return self._result

    def parse_file(self, file_name: str):
        """
        Parses the specified file and returns its representation as dictionary.
        :param file_name: file name to parse.
        :return: dictionary with file content or None if the file cannot be parsed.
        """
        if not os.path.exists(file_name):
            log.error('File {} does not exist'.format(file_name))
            return None
        try:
            with open(file_name) as file:
                file_content = file.readlines()
        except Exception as ex:
            log.error('Failed to read file {}: {}'.format(file_name, str(ex)))
            return None
        return self.parse_from_string(''.join(file_content))
