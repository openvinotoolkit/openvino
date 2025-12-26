import os

import yaml

from conan.internal.cache.home_paths import HomePaths
from conan.internal.default_settings import default_settings_yml
from conan.internal.internal_tools import is_universal_arch
from conan.errors import ConanException
from conan.internal.util.files import save, load


def bad_value_msg(name, value, value_range):
    return ("Invalid setting '%s' is not a valid '%s' value.\nPossible values are %s\n"
            'Read "http://docs.conan.io/2/knowledge/faq.html#error-invalid-setting"'
            # value range can be either a list or a dict, we only want to list the keys
            % (value, name, [v for v in value_range if v is not None]))


def undefined_field(name, field, fields=None, value=None):
    value_str = " for '%s'" % value if value else ""
    result = ["'%s.%s' doesn't exist%s" % (name, field, value_str),
              "'%s' possible configurations are %s" % (name, fields or "none")]
    return ConanException("\n".join(result))


class SettingsItem:
    """ represents a setting value and its child info, which could be:
    - A range of valid values: [Debug, Release] (for settings.compiler.runtime of VS)
    - List [None, "ANY"] to accept None or any value
    - A dict {subsetting: definition}, e.g. {version: [], runtime: []} for VS
    """
    def __init__(self, definition, name, value):
        self._definition = definition  # range of possible values
        self._name = name  # settings.compiler
        self._value = value  # gcc

    @staticmethod
    def new(definition, name):
        if definition is None:
            raise ConanException(f"Definition of settings.yml '{name}' cannot be null")
        if isinstance(definition, dict):
            parsed_definitions = {}
            # recursive
            for k, v in definition.items():
                # None string from yaml definition maps to python None, means not-defined value
                k = str(k) if k is not None else None
                parsed_definitions[k] = Settings(v, name, k)
        else:
            # list or tuple of possible values, it can include "ANY"
            parsed_definitions = [str(v) if v is not None else None for v in definition]
        return SettingsItem(parsed_definitions, name, None)

    def __contains__(self, value):
        return value in (self._value or "")

    def copy(self):
        """ deepcopy, recursive
        """
        if not isinstance(self._definition, dict):
            definition = self._definition  # Not necessary to copy this, not mutable
        else:
            definition = {k: v.copy() for k, v in self._definition.items()}
        return SettingsItem(definition, self._name, self._value)

    def copy_conaninfo_settings(self):
        """ deepcopy, recursive
        This function adds "ANY" to lists, to allow the ``package_id()`` method to modify some of
        values, but not all, just the "final" values without subsettings.
        We cannot let users manipulate to random strings
        things that contain subsettings like ``compiler``, because that would leave the thing
        in an undefined state, with some now inconsistent subsettings, that cannot be accessed
        anymore. So with this change the options are:
        - If you need more "binary-compatible" descriptions of a compiler, lets say like
        "gcc_or_clang", then you need to add that string to settings.yml. And add the subsettings
        that you want for it.
        - Settings that are "final" (lists), like build_type, or arch or compiler.version they
        can get any value without issues.
        """
        if not isinstance(self._definition, dict):
            definition = self._definition[:] + ["ANY"]
        else:
            definition = {k: v.copy_conaninfo_settings() for k, v in self._definition.items()}
            definition["ANY"] = Settings()
        return SettingsItem(definition, self._name, self._value)

    def __bool__(self):
        if not self._value:
            return False
        return self._value.lower() not in ["false", "none", "0", "off"]

    def __str__(self):
        return str(self._value)

    def __eq__(self, other):
        if other is None:
            return self._value is None
        other = self._validate(other)
        return other == self._value

    def __delattr__(self, item):
        """ This is necessary to remove libcxx subsetting from compiler in config()
           del self.settings.compiler.stdlib
        """
        child_setting = self._get_child(self._value)
        delattr(child_setting, item)

    def _validate(self, value):
        value = str(value) if value is not None else None
        is_universal = is_universal_arch(value, self._definition) if self._name == "settings.arch" else False
        if "ANY" not in self._definition and value not in self._definition and not is_universal:
            raise ConanException(bad_value_msg(self._name, value, self._definition))
        return value

    def _get_child(self, item):
        if not isinstance(self._definition, dict):
            raise undefined_field(self._name, item, None, self._value)
        if self._value is None:
            raise ConanException("'%s' value not defined" % self._name)
        return self._get_definition()

    def _get_definition(self):
        if self._value not in self._definition and "ANY" in self._definition:
            return self._definition["ANY"]
        return self._definition[self._value]

    def __getattr__(self, item):
        item = str(item)
        sub_config_dict = self._get_child(item)
        return getattr(sub_config_dict, item)

    def __setattr__(self, item, value):
        if item[0] == "_" or item.startswith("value"):
            return super(SettingsItem, self).__setattr__(item, value)

        item = str(item)
        sub_config_dict = self._get_child(item)
        return setattr(sub_config_dict, item, value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = self._validate(v)

    @property
    def values_range(self):
        # This needs to support 2 operations: "in" and iteration. Beware it can return "ANY"
        return self._definition

    @property
    def values_list(self):
        if self._value is None:
            return []
        result = []
        partial_name = ".".join(self._name.split(".")[1:])
        result.append((partial_name, self._value))
        if isinstance(self._definition, dict):
            sub_config_dict = self._get_definition()
            result.extend(sub_config_dict.values_list)
        return result

    def validate(self):
        if self._value is None and None not in self._definition:
            raise ConanException("'%s' value not defined" % self._name)
        if isinstance(self._definition, dict):
            self._get_definition().validate()

    def possible_values(self):
        if isinstance(self._definition, list):
            return self.values_range.copy()
        ret = {}
        for key, value in self._definition.items():
            ret[key] = value.possible_values()
        return ret

    def rm_safe(self, name):
        """ Iterates all possible subsettings, calling rm_safe() for all of them. If removing
        "compiler.cppstd", this will iterate msvc, gcc, clang, etc, calling rm_safe(cppstd) for
        all of them"""
        if isinstance(self._definition, list):
            return
        for subsetting in self._definition.values():
            subsetting.rm_safe(name)


class Settings:
    def __init__(self, definition=None, name="settings", parent_value="settings"):
        if parent_value is None and definition:
            raise ConanException("settings.yml: null setting can't have subsettings")
        definition = definition or {}
        if not isinstance(definition, dict):
            val = "" if parent_value == "settings" else f"={parent_value}"
            raise ConanException(f"Invalid settings.yml format: '{name}{val}' is not a dictionary")
        self._name = name  # settings, settings.compiler
        self._parent_value = parent_value  # gcc, x86
        self._data = {k: SettingsItem.new(v, f"{name}.{k}") for k, v in definition.items()}
        self._frozen = False

    def serialize(self):
        """
        Returns a dictionary with all the settings (and sub-settings) as ``field: value``
        """
        ret = []
        for _, s in self._data.items():
            # TODO: Refactor it and use s.serialize()
            ret.extend(s.values_list)
        return dict(ret)

    def get_safe(self, name, default=None):
        """
        Get the setting value avoiding throwing if it does not exist or has been removed
        :param name:
        :param default:
        :return:
        """
        try:
            tmp = self
            for prop in name.split("."):
                tmp = getattr(tmp, prop, None)
        except ConanException:
            return default
        if tmp is not None and tmp.value is not None:  # In case of subsettings is None
            return tmp.value
        return default

    def rm_safe(self, name):
        """ Removes the setting or subsetting from the definition. For example,
        rm_safe("compiler.cppstd") remove all "cppstd" subsetting from all compilers, irrespective
        of the current value of the "compiler"
        """
        if "." in name:
            setting, remainder = name.split(".", 1)  # setting=compiler, remainder = cppstd
            try:
                self._data[setting].rm_safe(remainder)  # call rm_safe("cppstd") for the "compiler"
            except KeyError:
                pass
        else:
            if name == "*":
                self.clear()
            else:
                self._data.pop(name, None)

    def copy(self):
        """ deepcopy, recursive
        """
        result = Settings({}, name=self._name, parent_value=self._parent_value)
        result._data = {k: v.copy() for k, v in self._data.items()}
        return result

    def copy_conaninfo_settings(self):
        result = Settings({}, name=self._name, parent_value=self._parent_value)
        result._data = {k: v.copy_conaninfo_settings() for k, v in self._data.items()}
        return result

    @staticmethod
    def loads(text):
        try:
            return Settings(yaml.safe_load(text) or {})
        except (yaml.YAMLError, AttributeError) as ye:
            raise ConanException("Invalid settings.yml format: {}".format(ye))

    def validate(self):
        for child in self._data.values():
            child.validate()

    @property
    def fields(self):
        return sorted(list(self._data.keys()))

    def clear(self):
        self._data = {}

    def _check_field(self, field):
        if field not in self._data:
            raise undefined_field(self._name, field, self.fields, self._parent_value)

    def __getattr__(self, field):
        assert field[0] != "_", "ERROR %s" % field
        self._check_field(field)
        return self._data[field]

    def __delattr__(self, field):
        assert field[0] != "_", "ERROR %s" % field
        self._check_field(field)
        del self._data[field]

    def __setattr__(self, field, value):
        if field[0] == "_":
            return super(Settings, self).__setattr__(field, value)

        self._check_field(field)
        if self._frozen:
            raise ConanException(f"Tried to define '{field}' setting inside recipe")
        self._data[field].value = value

    @property
    def values_list(self):
        # TODO: make it private, leave .items accessor only
        result = []
        for field in self.fields:
            config_item = self._data[field]
            result.extend(config_item.values_list)
        return result

    def items(self):
        return self.values_list

    def update_values(self, values, raise_undefined=True):
        """
        Receives a list of tuples (compiler.version, value)
        This is more an updater than a setter.
        """
        self._frozen = False  # Could be restored at the end, but not really necessary
        assert isinstance(values, (list, tuple)), values
        for (name, value) in values:
            list_settings = name.split(".")
            attr = self
            try:
                for setting in list_settings[:-1]:
                    attr = getattr(attr, setting)
                value = str(value) if value is not None else None
                setattr(attr, list_settings[-1], value)
            except ConanException:  # fails if receiving settings doesn't have it defined
                if raise_undefined:
                    raise

    def constrained(self, constraint_def):
        """ allows to restrict a given Settings object with the input of another Settings object
        1. The other Settings object MUST be exclusively a subset of the former.
           No additions allowed
        2. If the other defines {"compiler": None} means to keep the full specification
        """
        constraint_def = constraint_def or []
        if not isinstance(constraint_def, (list, tuple, set)):
            raise ConanException("Please defines settings as a list or tuple")

        for field in constraint_def:
            self._check_field(field)

        to_remove = [k for k in self._data if k not in constraint_def]
        for k in to_remove:
            del self._data[k]

    def dumps(self):
        """ produces a text string with lines containing a flattened version:
        compiler.arch = XX
        compiler.arch.speed = YY
        """
        result = []
        for (name, value) in self.values_list:
            # It is important to discard None values, so migrations in settings can be done
            # without breaking all existing packages SHAs, by adding a first None option
            # that doesn't change the final sha
            if value is not None:
                result.append("%s=%s" % (name, value))
        return '\n'.join(result)

    def possible_values(self):
        """Check the range of values of the definition of a setting
        """
        ret = {}
        for key, element in self._data.items():
            ret[key] = element.possible_values()
        return ret


def load_settings_yml(home_folder):
    """Returns {setting: [value, ...]} defining all the possible
               settings without values"""
    _home_paths = HomePaths(home_folder)
    settings_path = _home_paths.settings_path
    if not os.path.exists(settings_path):
        save(settings_path, default_settings_yml)
        save(settings_path + ".orig", default_settings_yml)  # stores a copy, to check migrations

    def _load_settings(path):
        try:
            return yaml.safe_load(load(path)) or {}
        except yaml.YAMLError as ye:
            raise ConanException("Invalid settings.yml format: {}".format(ye))

    settings = _load_settings(settings_path)
    user_settings_file = _home_paths.settings_path_user
    if os.path.exists(user_settings_file):
        settings_user = _load_settings(user_settings_file)

        def appending_recursive_dict_update(d, u):
            # Not the same behavior as conandata_update, because this append lists
            for k, v in u.items():
                if isinstance(v, list):
                    current = d.get(k) or []
                    d[k] = current + [value for value in v if value not in current]
                elif isinstance(v, dict):
                    current = d.get(k) or {}
                    if isinstance(current, list):  # convert to dict lists
                        current = {k: None for k in current}
                    d[k] = appending_recursive_dict_update(current, v)
                else:
                    d[k] = v
            return d

        appending_recursive_dict_update(settings, settings_user)

    return Settings(settings)
