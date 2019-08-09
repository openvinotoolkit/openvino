"""
Copyright (c) 2019 Intel Corporation

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

import enum
import math
import re
import warnings
from collections import OrderedDict
from copy import copy
from functools import partial
from pathlib import Path

from ..utils import get_path, string_to_bool


class ConfigError(ValueError):
    pass


class BaseValidator:
    def __init__(self, on_error=None, additional_validator=None):
        self.on_error = on_error
        self.additional_validator = additional_validator

        self.field_uri = None

    def validate(self, entry, field_uri=None):
        field_uri = field_uri or self.field_uri
        if self.additional_validator and not self.additional_validator(entry, field_uri):
            self.raise_error(entry, field_uri)

    def raise_error(self, value, field_uri, reason=None):
        if self.on_error:
            self.on_error(value, field_uri, reason)

        error_message = 'Invalid value "{value}" for {field_uri}'.format(value=value, field_uri=field_uri)
        if reason:
            error_message = '{error_message}: {reason}'.format(error_message=error_message, reason=reason)

        raise ConfigError(error_message.format(value, field_uri))


class _ExtraArgumentBehaviour(enum.Enum):
    WARN = 'warn'
    IGNORE = 'ignore'
    ERROR = 'error'


def _is_dict_like(entry):
    return hasattr(entry, '__iter__') and hasattr(entry, '__getitem__')


class ConfigValidator(BaseValidator):
    WARN_ON_EXTRA_ARGUMENT = _ExtraArgumentBehaviour.WARN
    ERROR_ON_EXTRA_ARGUMENT = _ExtraArgumentBehaviour.ERROR
    IGNORE_ON_EXTRA_ARGUMENT = _ExtraArgumentBehaviour.IGNORE

    def __init__(self, config_uri, on_extra_argument=WARN_ON_EXTRA_ARGUMENT, **kwargs):
        super().__init__(**kwargs)
        self.on_extra_argument = on_extra_argument

        self.fields = OrderedDict()
        self.field_uri = config_uri
        for name in dir(self):
            value = getattr(self, name)
            if not isinstance(value, BaseField):
                continue

            field_copy = copy(value)
            field_copy.field_uri = "{}.{}".format(config_uri, name)
            self.fields[name] = field_copy

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        field_uri = field_uri or self.field_uri
        if not _is_dict_like(entry):
            raise ConfigError("{} is expected to be dict-like".format(field_uri))

        extra_arguments = []
        for key in entry:
            if key not in self.fields:
                extra_arguments.append(key)
                continue

            self.fields[key].validate(entry[key])

        required_fields = set(name for name, value in self.fields.items() if not value.optional)
        missing_arguments = required_fields.difference(entry)

        if missing_arguments:
            arguments = ', '.join(map(str, missing_arguments))
            self.raise_error(
                entry, field_uri, "Invalid config for {}: missing required fields: {}".format(field_uri, arguments)
            )

        if extra_arguments:
            unknown_options_error = "specifies unknown options: {}".format(extra_arguments)
            message = "{} {}".format(field_uri, unknown_options_error)

            if self.on_extra_argument == _ExtraArgumentBehaviour.WARN:
                warnings.warn(message)
            if self.on_extra_argument == _ExtraArgumentBehaviour.ERROR:
                self.raise_error(entry, field_uri, message)

    @property
    def known_fields(self):
        return set(self.fields)

    def raise_error(self, value, field_uri, reason=None):
        if self.on_error:
            self.on_error(value, field_uri, reason)
        else:
            raise ConfigError(reason)


class BaseField(BaseValidator):
    def __init__(self, optional=False, allow_none=False, description=None, **kwargs):
        super().__init__(**kwargs)
        self.optional = optional
        self.allow_none = allow_none
        self.description = description

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        field_uri = field_uri or self.field_uri
        if not self.allow_none and entry is None:
            raise ConfigError("{} is not allowed to be None".format(field_uri))

    @property
    def type(self):
        return str


class StringField(BaseField):
    def __init__(self, choices=None, regex=None, case_sensitive=False, **kwargs):
        super().__init__(**kwargs)
        self.choices = choices if case_sensitive or not choices else list(map(str.lower, choices))
        self.regex = re.compile(regex, flags=re.IGNORECASE if not case_sensitive else 0) if regex else None
        self.case_sensitive = case_sensitive

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        source_entry = entry

        if not isinstance(entry, str):
            raise ConfigError("{} is expected to be str".format(source_entry))

        if not self.case_sensitive:
            entry = entry.lower()

        if self.choices and entry not in self.choices:
            reason = "unsupported option, expected one of: {}".format(', '.join(map(str, self.choices)))
            self.raise_error(source_entry, field_uri, reason)

        if self.regex and not self.regex.match(entry):
            self.raise_error(source_entry, field_uri, reason=None)

    @property
    def type(self):
        return str


class DictField(BaseField):
    def __init__(self, key_type=None, value_type=None, validate_keys=True, validate_values=True, allow_empty=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.validate_keys = validate_keys if key_type else False
        self.validate_values = validate_values if value_type else False
        self.key_type = _get_field_type(key_type)
        self.value_type = _get_field_type(value_type)

        self.allow_empty = allow_empty

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        if not isinstance(entry, dict):
            raise ConfigError("{} is expected to be dict".format(field_uri))

        if not entry and not self.allow_empty:
            self.raise_error(entry, field_uri, "value is empty")

        for k, v in entry.items():
            if self.validate_keys:
                uri = "{}.keys.{}".format(field_uri, k)
                self.key_type.validate(k, uri)

            if self.validate_values:
                uri = "{}.{}".format(field_uri, k)

                self.value_type.validate(v, uri)
    @property
    def type(self):
        return dict


class ListField(BaseField):
    def __init__(self, value_type=None, validate_values=True, allow_empty=True, **kwargs):
        super().__init__(**kwargs)
        self.validate_values = validate_values if value_type else False
        self.value_type = _get_field_type(value_type)
        self.allow_empty = allow_empty

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        if not isinstance(entry, list):
            raise ConfigError("{} is expected to be list".format(field_uri))

        if not entry and not self.allow_empty:
            self.raise_error(entry, field_uri, "value is empty")

        if self.validate_values:
            for i, val in enumerate(entry):
                self.value_type.validate(val, "{}[{}]".format(val, i))

    @property
    def type(self):
        return list


class NumberField(BaseField):
    def __init__(self, floats=True, min_value=None, max_value=None, allow_inf=False, allow_nan=False, **kwargs):
        super().__init__(**kwargs)
        self.floats = floats
        self.min = min_value
        self.max = max_value
        self.allow_inf = allow_inf
        self.allow_nan = allow_nan

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        if not self.floats and isinstance(entry, float):
            raise ConfigError("{} is expected to be int".format(field_uri))
        if not isinstance(entry, int) and not isinstance(entry, float):
            raise ConfigError("{} is expected to be number".format(field_uri))

        if self.min is not None and entry < self.min:
            reason = "value is less than minimal allowed - {}".format(self.min)
            self.raise_error(entry, field_uri, reason)
        if self.max is not None and entry > self.max:
            reason = "value is greater than maximal allowed - {}".format(self.max)
            self.raise_error(entry, field_uri, reason)

        if math.isinf(entry) and not self.allow_inf:
            self.raise_error(entry, field_uri, "value is infinity")
        if math.isnan(entry) and not self.allow_nan:
            self.raise_error(entry, field_uri, "value is NaN")

    @property
    def type(self):
        return float if self.floats else int


class PathField(BaseField):
    def __init__(self, is_directory=False, check_exists=True, **kwargs):
        super().__init__(**kwargs)
        self.is_directory = is_directory
        self.check_exists = check_exists

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        try:
            get_path(entry, self.is_directory, self.check_exists)
        except TypeError:
            self.raise_error(entry, field_uri, "values is expected to be path-like")
        except FileNotFoundError:
            self.raise_error(entry, field_uri, "path does not exist")
        except NotADirectoryError:
            self.raise_error(entry, field_uri, "path is not a directory")
        except IsADirectoryError:
            self.raise_error(entry, field_uri, "path is a directory, regular file expected")

    @property
    def type(self):
        return Path


class BoolField(BaseField):
    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        if not isinstance(entry, bool):
            raise ConfigError("{} is expected to be bool".format(field_uri))

    @property
    def type(self):
        return string_to_bool


def _get_field_type(key_type):
    if not isinstance(key_type, BaseField):
        type_ = _TYPE_TO_FIELD_CLASS.get(key_type)
        if callable(type_):
            return type_()

    return key_type


_TYPE_TO_FIELD_CLASS = {
    int: partial(NumberField, floats=False),
    float: partial(NumberField, floats=True),
    dict: partial(DictField, validate_keys=False, validate_values=False),
    list: partial(ListField, validate_values=False),
    Path: PathField,
    str: StringField,
    bool: BoolField,
}
