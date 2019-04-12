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

from math import inf, nan
from pathlib import Path
from unittest.mock import ANY

import pytest
from accuracy_checker.config.config_validator import (
    ConfigError,
    ConfigValidator,
    DictField,
    ListField,
    NumberField,
    PathField,
    StringField
)
from tests.common import mock_filesystem


class TestStringField:
    def test_expects_string(self):
        string_field = StringField()

        with pytest.raises(ConfigError):
            string_field.validate(b"foo")
        with pytest.raises(ConfigError):
            string_field.validate({})
        with pytest.raises(ConfigError):
            string_field.validate(42)

        string_field.validate("foo")

    def test_choices(self):
        string_field = StringField(choices=['foo', 'bar'])

        with pytest.raises(ConfigError):
            string_field.validate('baz')

        string_field.validate('bar')

    def test_case_sensitive(self):
        string_field = StringField(choices=['foo', 'bar'], case_sensitive=False)

        string_field.validate('foo')
        string_field.validate('FOO')

        string_field = StringField(choices=['foo', 'bar'], case_sensitive=True)

        string_field.validate('foo')
        with pytest.raises(ConfigError):
            string_field.validate('FOO')

    def test_regex(self):
        string_field = StringField(regex=r'foo\d*')

        string_field.validate('foo')
        string_field.validate('foo42')

        with pytest.raises(ConfigError):
            string_field.validate('baz')

    def test_custom_exception(self, mocker):
        stub = mocker.stub(name='custom_on_error')
        string_field = StringField(choices=['foo'], on_error=stub)

        with pytest.raises(ConfigError):
            string_field.validate('bar', 'foo')
        stub.assert_called_once_with('bar', 'foo', ANY)

    def test_custom_validator(self, mocker):
        stub = mocker.stub(name='custom_validator')
        string_field = StringField(choices=['foo'], additional_validator=stub)

        string_field.validate('foo', 'baz')
        stub.assert_called_once_with('foo', 'baz')


class TestNumberField:
    def test_expects_number(self):
        number_field = NumberField(floats=True)

        number_field.validate(1.0)
        with pytest.raises(ConfigError):
            number_field.validate("foo")
        with pytest.raises(ConfigError):
            number_field.validate({})
        with pytest.raises(ConfigError):
            number_field.validate([])

        number_field = NumberField(floats=False)
        number_field.validate(1)
        with pytest.raises(ConfigError):
            number_field.validate(1.0)

    def test_nans(self):
        number_field = NumberField(allow_nan=True)
        number_field.validate(nan)

        number_field = NumberField(allow_nan=False)
        with pytest.raises(ConfigError):
            number_field.validate(nan)

    def test_infinity(self):
        number_field = NumberField(allow_inf=True)
        number_field.validate(inf)

        number_field = NumberField(allow_inf=False)
        with pytest.raises(ConfigError):
            number_field.validate(inf)

    def test_ranges(self):
        number_field = NumberField(min_value=0, max_value=5)

        number_field.validate(0)
        number_field.validate(1)
        number_field.validate(2)

        with pytest.raises(ConfigError):
            number_field.validate(-1)
        with pytest.raises(ConfigError):
            number_field.validate(7)


class TestDictField:
    def test_expects_dict(self):
        dict_field = DictField()

        dict_field.validate({})
        with pytest.raises(ConfigError):
            dict_field.validate("foo")
        with pytest.raises(ConfigError):
            dict_field.validate(42)
        with pytest.raises(ConfigError):
            dict_field.validate([])

    def test_validates_keys(self):
        dict_field = DictField()
        dict_field.validate({'foo': 42, 1: 'bar'})

        dict_field = DictField(key_type=str)
        dict_field.validate({'foo': 42, 'bar': 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 1: 'bar'})

        dict_field = DictField(key_type=StringField(choices=['foo', 'bar']))
        dict_field.validate({'foo': 42, 'bar': 42})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 1: 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 'baz': 42})

    def test_validates_values(self):
        dict_field = DictField()
        dict_field.validate({'foo': 42, 1: 'bar'})

        dict_field = DictField(value_type=str)
        dict_field.validate({'foo': 'foo', 1: 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 1: 2})

        dict_field = DictField(value_type=StringField(choices=['foo', 'bar']))
        dict_field.validate({1: 'foo', 'bar': 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({1: 'foo', 2: 3})
        with pytest.raises(ConfigError):
            dict_field.validate({1: 'foo', 2: 'baz'})

    def test_converts_basic_types(self):
        dict_field = DictField(value_type=str)
        assert isinstance(dict_field.value_type, StringField)

        dict_field = DictField(value_type=int)
        assert isinstance(dict_field.value_type, NumberField)
        assert dict_field.value_type.floats is False

        dict_field = DictField(value_type=float)
        assert isinstance(dict_field.value_type, NumberField)
        assert dict_field.value_type.floats is True

        dict_field = DictField(value_type=list)
        assert isinstance(dict_field.value_type, ListField)

        dict_field = DictField(value_type=dict)
        assert isinstance(dict_field.value_type, DictField)

        dict_field = DictField(value_type=Path)
        assert isinstance(dict_field.value_type, PathField)

    def test_empty(self):
        dict_field = DictField()
        dict_field.validate({})

        dict_field = DictField(allow_empty=False)
        with pytest.raises(ConfigError):
            dict_field.validate({})


class TestListField:
    def test_expects_list(self):
        list_field = ListField()

        list_field.validate([])
        with pytest.raises(ConfigError):
            list_field.validate("foo")
        with pytest.raises(ConfigError):
            list_field.validate(42)
        with pytest.raises(ConfigError):
            list_field.validate({})

    def test_validates_values(self):
        list_field = ListField()
        list_field.validate(['foo', 42])

        list_field = ListField(value_type=str)
        list_field.validate(['foo', 'bar'])
        with pytest.raises(ConfigError):
            list_field.validate(['foo', 42])

        list_field = ListField(value_type=StringField(choices=['foo', 'bar']))
        list_field.validate(['foo', 'bar'])
        with pytest.raises(ConfigError):
            list_field.validate(['foo', 42])
        with pytest.raises(ConfigError):
            list_field.validate(['foo', 'bar', 'baz'])

    def test_empty(self):
        list_field = ListField()
        list_field.validate([])

        list_field = ListField(allow_empty=False)
        with pytest.raises(ConfigError):
            list_field.validate([])


class TestPathField:
    @pytest.mark.usefixtures('mock_path_exists')
    def test_expects_path_like(self):
        path_field = PathField()
        path_field.validate('foo/bar')
        path_field.validate('/home/user')
        path_field.validate(Path('foo/bar'))

        with pytest.raises(ConfigError):
            path_field.validate(42)
        with pytest.raises(ConfigError):
            path_field.validate({})
        with pytest.raises(ConfigError):
            path_field.validate([])

    def test_path_is_checked(self):
        with mock_filesystem(['foo/bar']) as prefix:
            prefix_path = Path(prefix)
            file_field = PathField(is_directory=False)
            with pytest.raises(ConfigError):
                file_field.validate(prefix_path / 'foo')
            file_field.validate(prefix_path / 'foo' / 'bar')

            dir_field = PathField(is_directory=True)
            dir_field.validate(prefix_path / 'foo')

            with pytest.raises(ConfigError):
                dir_field.validate(prefix_path / 'foo' / 'bar')


class TestConfigValidator:
    def test_compound(self):
        class SampleValidator(ConfigValidator):
            foo = StringField(choices=['foo'])
            bar = NumberField()

        sample_validator = SampleValidator('Sample')
        sample_validator.validate({'foo': 'foo', 'bar': 1})

        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'foo'})
        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'bar', 'bar': 1})

    def test_optional_fields(self):
        class SampleValidatorNoOptionals(ConfigValidator):
            foo = StringField(choices=['foo'])
            bar = NumberField(optional=False)

        sample_validator = SampleValidatorNoOptionals('Sample')
        sample_validator.validate({'foo': 'foo', 'bar': 1})
        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'bar'})

        class SampleValidatorWithOptionals(ConfigValidator):
            foo = StringField(choices=['foo'])
            bar = NumberField(optional=True)

        sample_validator = SampleValidatorWithOptionals('Sample')
        sample_validator.validate({'foo': 'foo', 'bar': 1})
        sample_validator.validate({'foo': 'foo'})

    def test_extra_fields__warn_on_extra(self):
        class SampleValidatorWarnOnExtra(ConfigValidator):
            foo = StringField(choices=['foo'])

        sample_validator = SampleValidatorWarnOnExtra(
            'Sample', on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

        with pytest.warns(UserWarning):
            sample_validator.validate({'foo': 'foo', 'bar': 'bar'})

    def test_extra_fields__error_on_extra(self):
        class SampleValidatorErrorOnExtra(ConfigValidator):
            foo = StringField(choices=['foo'])

        sample_validator = SampleValidatorErrorOnExtra(
            'Sample', on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'bar', 'bar': 'bar'})

    def test_extra_fields__ignore_extra(self):
        class SampleValidatorIgnoresExtra(ConfigValidator):
            foo = StringField(choices=['foo'])

        sample_validator = SampleValidatorIgnoresExtra(
            'Sample', on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

        sample_validator.validate({'foo': 'foo', 'bar': 'bar'})

    def test_custom_exception(self, mocker):
        class SampleValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        stub = mocker.stub(name='custom_on_error')
        sample_validator = SampleValidator('Sample', on_error=stub)
        sample_validator.validate({})
        stub.assert_called_once_with(ANY, 'Sample', ANY)

    def test_custom_validator(self, mocker):
        class SampleValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        stub = mocker.stub(name='custom_validator')
        sample_validator = SampleValidator('Sample', additional_validator=stub)
        entry = {'foo': 'foo'}
        sample_validator.validate(entry)
        stub.assert_called_once_with(entry, 'Sample')

    def test_nested(self):
        class InnerValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        class OuterValidator(ConfigValidator):
            bar = ListField(InnerValidator('Inner'))

        outer_validator = OuterValidator('Outer', on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

        outer_validator.validate({'bar': [{'foo': 'foo'}, {'foo': 'foo'}]})

    def test_inheritance(self):
        class ParentValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        class DerivedValidator(ParentValidator):
            bar = StringField(choices=['bar'])

        derived_validator = DerivedValidator('Derived', on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)
        derived_validator.validate({'foo': 'foo', 'bar': 'bar'})
