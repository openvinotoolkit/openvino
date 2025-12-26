import hashlib

from conan.errors import ConanException
from conan.internal.model.dependencies import UserRequirementsDict
from conan.api.model import PkgReference
from conan.api.model import RecipeReference
from conan.internal.util.config_parser import TextINIParse


class _VersionRepr:
    """Class to return strings like 1.Y.Z from a Version object"""

    def __init__(self, version):
        self._version = version

    def stable(self):
        if self._version.major == 0:
            return str(self._version)
        else:
            return self.major()

    def major(self):
        # This check is to avoid breaking non-integer major versions
        # for legacy reasons. Users are warned against using them
        if not isinstance(self._version.major.value, int):
            return str(self._version.major)
        return ".".join([str(self._version.major), 'Y', 'Z'])

    def minor(self):
        # This check is to avoid breaking non-integer major versions
        # for legacy reasons. Users are warned against using them
        if not isinstance(self._version.major.value, int):
            return str(self._version.major)

        v0 = str(self._version.major)
        v1 = str(self._version.minor) if self._version.minor is not None else "0"
        return ".".join([v0, v1, 'Z'])

    def patch(self):
        # This check is to avoid breaking non-integer major versions
        # for legacy reasons. Users are warned against using them
        if not isinstance(self._version.major.value, int):
            return str(self._version.major)

        v0 = str(self._version.major)
        v1 = str(self._version.minor) if self._version.minor is not None else "0"
        v2 = str(self._version.patch) if self._version.patch is not None else "0"
        return ".".join([v0, v1, v2])

    def pre(self):
        # This check is to avoid breaking non-integer major versions
        # for legacy reasons. Users are warned against using them
        if not isinstance(self._version.major.value, int):
            return str(self._version.major)

        v0 = str(self._version.major)
        v1 = str(self._version.minor) if self._version.minor is not None else "0"
        v2 = str(self._version.patch) if self._version.patch is not None else "0"
        v = ".".join([v0, v1, v2])
        if self._version.pre is not None:
            v += "-%s" % self._version.pre
        return v

    @property
    def build(self):
        return self._version.build if self._version.build is not None else ""


class RequirementInfo:

    def __init__(self, ref, package_id, default_package_id_mode):
        self._ref = ref
        self._package_id = package_id
        self.name = self.version = self.user = self.channel = self.package_id = None
        self.recipe_revision = None
        self.package_id_mode = default_package_id_mode

        try:
            func_package_id_mode = getattr(self, default_package_id_mode)
        except AttributeError:
            raise ConanException(f"require {self._ref} package_id_mode='{default_package_id_mode}' "
                                 "is not a known package_id_mode")
        else:
            func_package_id_mode()

    def copy(self):
        # Useful for build_id()
        result = RequirementInfo(self._ref, self._package_id, "unrelated_mode")
        for f in ("name", "version", "user", "channel", "recipe_revision", "package_id"):
            setattr(result, f, getattr(self, f))
        return result

    def pref(self):
        ref = RecipeReference(self.name, self.version, self.user, self.channel, self.recipe_revision)
        return PkgReference(ref, self.package_id)

    def dumps(self):
        return repr(self.pref())

    def unrelated_mode(self):
        self.name = self.version = self.user = self.channel = self.package_id = None
        self.recipe_revision = None

    def semver_mode(self):
        self.name = self._ref.name
        self.version = _VersionRepr(self._ref.version).stable()
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = None
        self.recipe_revision = None

    def full_version_mode(self):
        self.name = self._ref.name
        self.version = self._ref.version
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = None
        self.recipe_revision = None

    def patch_mode(self):
        self.name = self._ref.name
        self.version = _VersionRepr(self._ref.version).patch()
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = None
        self.recipe_revision = None

    def minor_mode(self):
        self.name = self._ref.name
        self.version = _VersionRepr(self._ref.version).minor()
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = None
        self.recipe_revision = None

    def major_mode(self):
        self.name = self._ref.name
        self.version = _VersionRepr(self._ref.version).major()
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = None
        self.recipe_revision = None

    def full_package_mode(self):
        self.name = self._ref.name
        self.version = self._ref.version
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = self._package_id
        self.recipe_revision = None

    def revision_mode(self):
        self.name = self._ref.name
        self.version = self._ref.version
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = None
        self.recipe_revision = self._ref.revision

    def full_mode(self):
        self.name = self._ref.name
        self.version = self._ref.version
        self.user = self._ref.user
        self.channel = self._ref.channel
        self.package_id = self._package_id
        self.recipe_revision = self._ref.revision

    full_recipe_mode = full_version_mode
    recipe_revision_mode = full_mode  # to not break everything and help in upgrade


class RequirementsInfo(UserRequirementsDict):

    def copy(self):
        # For build_id() implementation
        data = {pref: req_info.copy() for pref, req_info in self._data.items()}
        return RequirementsInfo(data)

    def serialize(self):
        return [r.dumps() for r in self._data.values()]

    def __bool__(self):
        return bool(self._data)

    def clear(self):
        self._data = {}

    def remove(self, *args):
        for name in args:
            del self[name]

    @property
    def pkg_names(self):
        return [r.ref.name for r in self._data.keys()]

    def dumps(self):
        result = []
        for req_info in self._data.values():
            dumped = req_info.dumps()
            if dumped:
                result.append(dumped)
        return "\n".join(sorted(result))

    def unrelated_mode(self):
        self.clear()

    def semver_mode(self):
        for r in self._data.values():
            r.semver_mode()

    def patch_mode(self):
        for r in self._data.values():
            r.patch_mode()

    def minor_mode(self):
        for r in self._data.values():
            r.minor_mode()

    def major_mode(self):
        for r in self._data.values():
            r.major_mode()

    def full_version_mode(self):
        for r in self._data.values():
            r.full_version_mode()

    def full_recipe_mode(self):
        for r in self._data.values():
            r.full_recipe_mode()

    def full_package_mode(self):
        for r in self._data.values():
            r.full_package_mode()

    def revision_mode(self):
        for r in self._data.values():
            r.revision_mode()

    def full_mode(self):
        for r in self._data.values():
            r.full_mode()

    recipe_revision_mode = full_mode  # to not break everything and help in upgrade


class PythonRequiresInfo:

    def __init__(self, refs, default_package_id_mode):
        self._default_package_id_mode = default_package_id_mode
        if refs:
            self._refs = [RequirementInfo(r, None,
                                          default_package_id_mode=mode or default_package_id_mode)
                          for r, mode in sorted(refs.items())]
        else:
            self._refs = None

    def copy(self):
        # For build_id() implementation
        refs = {r._ref: r.package_id_mode for r in self._refs} if self._refs else None
        return PythonRequiresInfo(refs, self._default_package_id_mode)

    def serialize(self):
        return [r.dumps() for r in self._refs or []]

    def __bool__(self):
        return bool(self._refs)

    def clear(self):
        self._refs = None

    def dumps(self):
        return '\n'.join(r.dumps() for r in self._refs)

    def unrelated_mode(self):
        self._refs = None

    def semver_mode(self):
        for r in self._refs:
            r.semver_mode()

    def patch_mode(self):
        for r in self._refs:
            r.patch_mode()

    def minor_mode(self):
        for r in self._refs:
            r.minor_mode()

    def major_mode(self):
        for r in self._refs:
            r.major_mode()

    def full_version_mode(self):
        for r in self._refs:
            r.full_version_mode()

    def full_recipe_mode(self):
        for r in self._refs:
            r.full_recipe_mode()

    def revision_mode(self):
        for r in self._refs:
            r.revision_mode()

    def full_mode(self):
        for r in self._refs:
            r.full_mode()

    recipe_revision_mode = full_mode


def load_binary_info(text):
    # This is used for search functionality, search prints info from this file
    parser = TextINIParse(text)
    conan_info_json = {}
    for section, lines in parser.line_items():
        try:
            items = [line.split("=", 1) for line in lines]
            conan_info_json[section] = {item[0].strip(): item[1].strip() for item in items}
        except IndexError:
            conan_info_json[section] = lines

    return conan_info_json


class ConanInfo:

    def __init__(self, settings=None, options=None, reqs_info=None, build_requires_info=None,
                 python_requires=None, conf=None, config_version=None):
        self.invalid = None
        self.cant_build = False  # It will set to a str with a reason if the validate_build() fails
        self.settings = settings
        self.settings_target = None  # needs to be explicitly defined by recipe package_id()
        self.options = options
        self.requires = reqs_info
        self.build_requires = build_requires_info
        self.python_requires = python_requires
        self.conf = conf
        self.config_version = config_version
        self.compatibility_delta = None

    def clone(self):
        """ Useful for build_id implementation and for compatibility()
        """
        result = ConanInfo()
        result.invalid = self.invalid
        result.settings = self.settings.copy()
        result.options = self.options.copy_conaninfo_options()
        result.requires = self.requires.copy()
        result.build_requires = self.build_requires.copy()
        result.python_requires = self.python_requires.copy()
        result.conf = self.conf.copy()
        result.settings_target = self.settings_target.copy() if self.settings_target else None
        result.config_version = self.config_version.copy() if self.config_version else None
        return result

    def serialize(self):
        result = {}
        settings_dumps = self.settings.serialize()
        if settings_dumps:
            result["settings"] = settings_dumps
        if self.settings_target is not None:
            settings_target_dumps = self.settings_target.serialize()
            if settings_target_dumps:
                result["settings_target"] = settings_target_dumps
        options_dumps = self.options.serialize()
        if options_dumps:
            result["options"] = options_dumps
        requires_dumps = self.requires.serialize()
        if requires_dumps:
            result["requires"] = requires_dumps
        python_requires_dumps = self.python_requires.serialize()
        if python_requires_dumps:
            result["python_requires"] = python_requires_dumps
        build_requires_dumps = self.build_requires.serialize()
        if build_requires_dumps:
            result["build_requires"] = build_requires_dumps
        conf_dumps = self.conf.serialize()
        if conf_dumps:
            result["conf"] = conf_dumps
        config_version_dumps = self.config_version.serialize() if self.config_version else None
        if config_version_dumps:
            result["config_version"] = config_version_dumps
        if self.compatibility_delta:
            result["compatibility_delta"] = self.compatibility_delta
        return result

    def dumps(self):
        """
        Get all the information contained in settings, options, requires,
        python_requires, build_requires and conf.
        :return: `str` with the result of joining all the information, e.g.,
            `"[settings]\nos=Windows\n[options]\nuse_Qt=True"`
        """
        result = []
        settings_dumps = self.settings.dumps()
        if settings_dumps:
            result.append("[settings]")
            result.append(settings_dumps)
        if self.settings_target:
            settings_target_dumps = self.settings_target.dumps()
            if settings_target_dumps:
                result.append("[settings_target]")
                result.append(settings_target_dumps)
        options_dumps = self.options.dumps()
        if options_dumps:
            result.append("[options]")
            result.append(options_dumps)
        requires_dumps = self.requires.dumps()
        if requires_dumps:
            result.append("[requires]")
            result.append(requires_dumps)
        if self.python_requires:
            python_reqs_dumps = self.python_requires.dumps()
            if python_reqs_dumps:
                result.append("[python_requires]")
                result.append(python_reqs_dumps)
        if self.build_requires:
            build_requires_dumps = self.build_requires.dumps()
            if build_requires_dumps:
                result.append("[build_requires]")
                result.append(build_requires_dumps)
        if self.conf:
            # TODO: Think about the serialization of Conf, not 100% sure if dumps() is the best
            result.append("[conf]")
            result.append(self.conf.dumps())
        config_version_dumps = self.config_version.dumps() if self.config_version else None
        if config_version_dumps:
            result.append("[config_version]")
            result.append(config_version_dumps)
        result.append("")  # Append endline so file ends with LF
        return '\n'.join(result)

    def summarize_compact(self):
        result = []
        serialized = self.serialize()

        for key, values in serialized.items():
            if isinstance(values, dict):
                result.append(f"{key}: " + " ".join(f"{k}={v}" for k, v in values.items()))
            elif isinstance(values, type([])):
                result.append(f"{key}: {' '.join(values)}")

        return result

    def dump_diff(self, compatible):
        self_dump = self.dumps()
        compatible_dump = compatible.dumps()
        result = []
        for line in compatible_dump.splitlines():
            if line not in self_dump:
                result.append(line)
        return ', '.join(result)

    def package_id(self):
        """
        Get the `package_id` that is the result of applying the has function SHA-1 to the
        `self.dumps()` return.
        :return: `str` the `package_id`, e.g., `"040ce2bd0189e377b2d15eb7246a4274d1c63317"`
        """
        text = self.dumps()
        md = hashlib.sha1()
        md.update(text.encode())
        package_id = md.hexdigest()
        return package_id

    def clear(self):
        self.settings.clear()
        self.options.clear()
        self.requires.clear()
        self.conf.clear()
        self.build_requires.clear()
        self.python_requires.clear()
        if self.config_version is not None:
            self.config_version.clear()

    def validate(self):
        # If the options are not fully defined, this is also an invalid case
        try:
            self.options.validate()
        except ConanException as e:
            self.invalid = str(e)

        try:
            self.settings.validate()
        except ConanException as e:
            self.invalid = str(e)
