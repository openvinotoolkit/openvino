from conan.errors import ConanException
from conan.internal.model.recipe_ref import ref_matches

_falsey_options = ["false", "none", "0", "off", ""]


def option_not_exist_msg(option_name, existing_options):
    """ Someone is referencing an option that is not available in the current package
    options
    """
    result = ["option '%s' doesn't exist" % option_name,
              "Possible options are %s" % existing_options or "none"]
    return "\n".join(result)


class _PackageOption:
    def __init__(self, name, value, possible_values=None):
        self._name = name
        self._value = value  # Value None = not defined
        self.important = False
        # possible_values only possible origin is recipes
        if possible_values is None:
            self._possible_values = None
        else:
            # This can contain "ANY"
            self._possible_values = [str(v) if v is not None else None for v in possible_values]

    def dumps(self, scope=None):
        if self._value is None:
            return None
        important = "!" if self.important else ""
        if scope:
            return "%s:%s%s=%s" % (scope, self._name, important, self._value)
        else:
            return "%s%s=%s" % (self._name, important, self._value)

    def copy_conaninfo_option(self):
        # To generate a copy without validation, for package_id info.options value
        assert self._possible_values is not None  # this should always come from recipe, with []
        return _PackageOption(self._name, self._value, self._possible_values + ["ANY"])

    def __bool__(self):
        if self._value is None:
            return False
        return self._value.lower() not in _falsey_options

    def __str__(self):
        return str(self._value)

    def __int__(self):
        return int(self._value)

    def _check_valid_value(self, value):
        """ checks that the provided value is allowed by current restrictions
        """
        if self._possible_values is None:  # validation not defined (profile)
            return
        if value in self._possible_values:
            return
        if value is not None and "ANY" in self._possible_values:
            return
        msg = ("'%s' is not a valid 'options.%s' value.\nPossible values are %s"
               % (value, self._name, self._possible_values))
        raise ConanException(msg)

    def __eq__(self, other):
        # To promote the other to string, and always compare as strings
        # if self.options.myoption == 1 => will convert 1 to "1"
        if other is None:
            return self._value is None
        other = str(other)
        self._check_valid_value(other)
        if self._value is None:
            return False  # Other is not None here
        return other == self.__str__()

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        v = str(v) if v is not None else None
        self._check_valid_value(v)
        self._value = v

    def validate(self):
        # check that this has a valid option value defined
        if self._value is not None:
            return
        if None not in self._possible_values:
            raise ConanException("'options.%s' value not defined" % self._name)


class _PackageOptions:
    def __init__(self, recipe_options_definition=None):
        if recipe_options_definition is None:
            self._constrained = False
            self._data = {}
        else:
            self._constrained = True
            self._data = {str(option): _PackageOption(str(option), None, possible_values)
                          for option, possible_values in recipe_options_definition.items()}
        self._freeze = False

    def dumps(self, scope=None):
        result = []
        for _, package_option in sorted(list(self._data.items())):
            dump = package_option.dumps(scope)
            if dump:
                result.append(dump)
        return "\n".join(result)

    @property
    def possible_values(self):
        return {k: v._possible_values for k, v in self._data.items()}

    def update(self, options):
        """
        @type options: _PackageOptions
        """
        # Necessary for init() extending of options for python_requires_extend
        for k, v in options._data.items():
            self._data[k] = v

    def clear(self):
        # for header_only() clearing
        self._data.clear()

    def freeze(self):
        self._freeze = True

    def __contains__(self, option):
        return str(option) in self._data

    def get_safe(self, field, default=None):
        return self._data.get(field, default)

    def rm_safe(self, field):
        # This should never raise any exception, in any case
        self._data.pop(field, None)

    def validate(self):
        for child in self._data.values():
            child.validate()

    def copy_conaninfo_options(self):
        # To generate a copy without validation, for package_id info.options value
        result = _PackageOptions()
        for k, v in self._data.items():
            result._data[k] = v.copy_conaninfo_option()
        return result

    def _ensure_exists(self, field):
        if self._constrained and field not in self._data:
            raise ConanException(option_not_exist_msg(field, list(self._data.keys())))

    def __getattr__(self, field):
        assert field[0] != "_", "ERROR %s" % field
        try:
            return self._data[field]
        except KeyError:
            raise ConanException(option_not_exist_msg(field, list(self._data.keys())))

    def __delattr__(self, field):
        assert field[0] != "_", "ERROR %s" % field
        # It is always possible to remove an option, even if it is frozen (freeze=True),
        # and it got a value, because it is the only way an option could be removed
        # conditionally to other option value (like fPIC if shared)
        self._ensure_exists(field)
        del self._data[field]

    def __setattr__(self, field, value):
        if field[0] == "_":
            return super(_PackageOptions, self).__setattr__(field, value)
        self._set(field, value)

    def __setitem__(self, item, value):
        self._set(item, value)

    def _set(self, item, value):
        # programmatic way to define values, for Conan codebase
        important = item[-1] == "!"
        item = item[:-1] if important else item

        current_value = self._data.get(item)
        if self._freeze and current_value.value is not None and current_value != value:
            raise ConanException(f"Incorrect attempt to modify option '{item}' "
                                 f"from '{current_value}' to '{value}'")
        self._ensure_exists(item)
        v = self._data.setdefault(item, _PackageOption(item, None))
        new_value_important = important or (isinstance(value, _PackageOption) and value.important)
        if new_value_important or not v.important:
            v.value = value
            v.important = new_value_important

    def items(self):
        result = []
        for field, package_option in sorted(list(self._data.items())):
            result.append((field, package_option.value))
        return result

    def update_options(self, other, is_pattern=False):
        """
        @param is_pattern: if True, then the value might not exist and won't be updated
        @type other: _PackageOptions
        """
        for k, v in other._data.items():
            if is_pattern and k not in self._data:
                continue
            self._set(k, v)


class Options:

    def __init__(self, options=None, options_values=None):
        # options=None means an unconstrained/profile definition
        try:
            self._package_options = _PackageOptions(options)
            # Addressed only by name, as only 1 configuration is allowed
            # if more than 1 is present, 1 should be "private" requirement and its options
            # are not public, not overridable
            self._deps_package_options = {}  # {name("Boost": PackageOptions}
            if options_values:
                for k, v in options_values.items():
                    if v is None:
                        continue  # defining a None value means same as not giving value
                    k = str(k).strip()
                    v = str(v).strip()
                    tokens = k.split(":", 1)
                    if len(tokens) == 2:
                        package, option = tokens
                        if not package:
                            raise ConanException("Invalid empty package name in options. "
                                                 f"Use a pattern like `mypkg/*:{option}`")
                        if "/" not in package and "*" not in package and "&" not in package:
                            msg = "The usage of package names `{}` in options is " \
                                  "deprecated, use a pattern like `{}/*:{}` " \
                                  "instead".format(k, package, option)
                            raise ConanException(msg)
                        self._deps_package_options.setdefault(package, _PackageOptions())[option] = v
                    else:
                        self._package_options[k] = v
        except Exception as e:
            raise ConanException("Error while initializing options. %s" % str(e))

    def __repr__(self):
        return self.dumps()

    @property
    def possible_values(self):
        return self._package_options.possible_values

    def dumps(self):
        """ produces a multiline text representation of all values, first self then others.
        In alphabetical order, skipping real None (not string "None") values:
            option1=value1
            other_option=3
            OtherPack:opt3=12.1
        """
        result = []
        pkg_options_dumps = self._package_options.dumps()
        if pkg_options_dumps:
            result.append(pkg_options_dumps)
        for pkg_pattern, pkg_option in sorted(self._deps_package_options.items()):
            dep_pkg_option = pkg_option.dumps(scope=pkg_pattern)
            if dep_pkg_option:
                result.append(dep_pkg_option)
        return "\n".join(result)

    @staticmethod
    def loads(text):
        """ parses a multiline text in the form produced by dumps(), NO validation here
        """
        values = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, value = line.split("=", 1)
                values[name] = value
            except ValueError:
                raise ConanException(f"Error while parsing option '{line}'. "
                                     f"Options should be specified as 'pkg/*:option=value'")
        return Options(options_values=values)

    def serialize(self):
        # used by ConanInfo serialization, involved in "list package-ids" output
        # we need to maintain the "options" and "req_options" first level or servers will break
        # This happens always after reading from conaninfo.txt => all str and not None
        result = {k: v for k, v in self._package_options.items()}
        # Include the dependencies ones, in case they have been explicitly added in package_id()
        # to the conaninfo.txt, we want to report them
        for pkg_pattern, pkg_option in sorted(self._deps_package_options.items()):
            for key, value in pkg_option.items():
                result["%s:%s" % (pkg_pattern, key)] = value
        return result

    def clear(self):
        # for header_only() clearing
        self._package_options.clear()
        self._deps_package_options.clear()

    def __contains__(self, option):
        return option in self._package_options

    def __getattr__(self, attr):
        return getattr(self._package_options, attr)

    def __setattr__(self, attr, value):
        if attr[0] == "_" or attr == "values":
            return super(Options, self).__setattr__(attr, value)
        return setattr(self._package_options, attr, value)

    def __delattr__(self, field):
        self._package_options.__delattr__(field)

    def __getitem__(self, item):
        if isinstance(item, str):
            if "/" not in item and "*" not in item:  # FIXME: To allow patterns like "*" or "foo*"
                item += "/*"
        return self._deps_package_options.setdefault(item, _PackageOptions())

    def scope(self, ref):
        """ when there are free options like "shared=True", they apply to the "consumer" package
        Once we know the name of such consumer package, it can be defined in the data, so it will
        be later correctly apply when processing options """
        package_options = self._deps_package_options.setdefault(str(ref), _PackageOptions())
        package_options.update_options(self._package_options)
        self._package_options = _PackageOptions()

    def copy_conaninfo_options(self):
        # To generate the package_id info.options copy, that can destroy, change and remove things
        result = Options()
        result._package_options = self._package_options.copy_conaninfo_options()
        # In most scenarios this should be empty at this stage, because it was cleared
        if self._deps_package_options:
            raise ConanException("Dependencies options were defined incorrectly. Maybe you"
                                 " tried to define options values in 'requirements()' or other"
                                 " invalid place")
        return result

    def update(self, options=None, options_values=None):
        # Necessary for init() extending of options for python_requires_extend
        new_options = Options(options, options_values)
        self._package_options.update(new_options._package_options)
        for pkg, pkg_option in new_options._deps_package_options.items():
            self._deps_package_options.setdefault(pkg, _PackageOptions()).update(pkg_option)

    def update_options(self, other):
        """
        dict-like update of options, "other" has priority, overwrite existing
        @type other: Options
        """
        self._package_options.update_options(other._package_options)
        for pkg, pkg_option in other._deps_package_options.items():
            self._deps_package_options.setdefault(pkg, _PackageOptions()).update_options(pkg_option)

    def apply_downstream(self, down_options, profile_options, own_ref, is_consumer):
        """ compute the current package options, starting from the self defined ones and applying
        the options defined by the downstrream consumers and the profile
        Only modifies the current package_options, not the dependencies ones
        """
        assert isinstance(down_options, Options)
        assert isinstance(profile_options, Options)

        for defined_options in down_options, profile_options:
            if own_ref is None or own_ref.name is None:
                # If the current package doesn't have a name defined, is a pure consumer without name
                # Get the non-scoped options, plus the "all-matching=*" pattern
                self._package_options.update_options(defined_options._package_options)
                for pattern, options in defined_options._deps_package_options.items():
                    if ref_matches(None, pattern, is_consumer=is_consumer):
                        self._package_options.update_options(options, is_pattern=True)
            else:
                # If the current package has a name, there should be a match, either exact name
                # match, or a fnmatch approximate one
                for pattern, options in defined_options._deps_package_options.items():
                    if ref_matches(own_ref, pattern, is_consumer=is_consumer):
                        self._package_options.update_options(options, is_pattern="*" in pattern)

        self._package_options.freeze()

    def get_upstream_options(self, down_options, own_ref, is_consumer):
        """ compute which options should be propagated to the dependencies, a combination of the
        downstream defined default_options with the current default_options ones. This happens
        at "configure()" time, while building the graph. Also compute the minimum "self_options"
        which is the state that a package should define in order to reproduce
        """
        assert isinstance(down_options, Options)
        # We need to store a copy for internal propagation for test_requires and tool_requires
        private_deps_options = Options()
        private_deps_options._deps_package_options = self._deps_package_options.copy()
        # self_options are the minimal necessary for a build-order
        # TODO: check this, isn't this just a copy?
        self_options = Options()
        self_options._deps_package_options = down_options._deps_package_options.copy()

        # compute now the necessary to propagate all down - self + self deps
        upstream_options = Options()
        for pattern, options in down_options._deps_package_options.items():
            if ref_matches(own_ref, pattern, is_consumer=is_consumer):
                # Remove the exact match-name to this package, don't further propagate up
                pattern_name = pattern.split("/", 1)[0]
                if "*" not in pattern_name:
                    continue
            self._deps_package_options.setdefault(pattern, _PackageOptions()).update_options(options)

        upstream_options._deps_package_options = self._deps_package_options
        # When the upstream is computed, the current dependencies are invalidated, so users will
        # not be able to do ``self.options["mydep"]`` because it will be empty. self.dependencies
        # is the way to access dependencies (in other methods)
        self._deps_package_options = {}
        return self_options, upstream_options, private_deps_options
