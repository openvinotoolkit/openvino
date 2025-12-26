import traceback
from importlib import invalidate_caches, util as imp_util
import inspect
import os
import re
import sys
import types
import uuid
from threading import Lock

import yaml

from pathlib import Path

from conan.api.output import ConanOutput
from conan.tools.cmake import cmake_layout
from conan.tools.google import bazel_layout
from conan.tools.microsoft import vs_layout
from conan.internal.errors import conanfile_exception_formatter, NotFoundException
from conan.errors import ConanException
from conan.internal.model.conan_file import ConanFile
from conan.internal.model.options import Options
from conan.api.model import RecipeReference
from conan.internal.paths import DATA_YML
from conan.internal.model.version_range import validate_conan_version
from conan.internal.util.config_parser import TextINIParse
from conan.internal.util.files import load, chdir, load_user_encoded


class ConanFileLoader:

    def __init__(self, pyreq_loader=None, conanfile_helpers=None):
        self._pyreq_loader = pyreq_loader
        self._cached_conanfile_classes = {}
        self._conanfile_helpers = conanfile_helpers
        invalidate_caches()

    def load_basic(self, conanfile_path, graph_lock=None, display="", remotes=None,
                   update=None, check_update=None):
        """ loads a conanfile basic object without evaluating anything
        """
        return self.load_basic_module(conanfile_path, graph_lock, display, remotes,
                                      update, check_update)[0]

    def load_basic_module(self, conanfile_path, graph_lock=None, display="", remotes=None,
                          update=None, check_update=None, tested_python_requires=None):
        """ loads a conanfile basic object without evaluating anything, returns the module too
        """
        cached = self._cached_conanfile_classes.get(conanfile_path)
        if cached:
            conanfile = cached[0](display)
            conanfile._conan_helpers = self._conanfile_helpers
            if hasattr(conanfile, "init") and callable(conanfile.init):
                with conanfile_exception_formatter(conanfile, "init"):
                    conanfile.init()
            return conanfile, cached[1]

        try:
            module, conanfile = _parse_conanfile(conanfile_path)
            if isinstance(tested_python_requires, RecipeReference):
                if getattr(conanfile, "python_requires", None) == "tested_reference_str":
                    conanfile.python_requires = tested_python_requires.repr_notime()
            elif tested_python_requires:
                if getattr(conanfile, "python_requires", None) != "tested_reference_str":
                    ConanOutput().warning("test_package/conanfile.py should declare 'python_requires"
                                          " = \"tested_reference_str\"'", warn_tag="deprecated")
                conanfile.python_requires = tested_python_requires

            if self._pyreq_loader:
                self._pyreq_loader.load_py_requires(conanfile, self, graph_lock, remotes,
                                                    update, check_update)

            conanfile.recipe_folder = os.path.dirname(conanfile_path)
            conanfile.recipe_path = Path(conanfile.recipe_folder)

            # Load and populate dynamic fields from the data file
            conan_data = self._load_data(conanfile_path)
            conanfile.conan_data = conan_data

            self._cached_conanfile_classes[conanfile_path] = (conanfile, module)
            result = conanfile(display)

            result._conan_helpers = self._conanfile_helpers
            if hasattr(result, "init") and callable(result.init):
                with conanfile_exception_formatter(result, "init"):
                    result.init()
            return result, module
        except ConanException as e:
            raise ConanException("Error loading conanfile at '{}': {}".format(conanfile_path, e))

    @staticmethod
    def _load_data(conanfile_path):
        data_path = os.path.join(os.path.dirname(conanfile_path), DATA_YML)
        if not os.path.exists(data_path):
            return None

        try:
            data = yaml.safe_load(load(data_path))
        except Exception as e:
            raise ConanException("Invalid yml format at {}: {}".format(DATA_YML, e))

        return data or {}

    def load_named(self, conanfile_path, name, version, user, channel, graph_lock=None,
                   remotes=None, update=None, check_update=None, tested_python_requires=None):
        """ loads the basic conanfile object and evaluates its name and version
        """
        conanfile, _ = self.load_basic_module(conanfile_path, graph_lock, remotes=remotes,
                                              update=update, check_update=check_update,
                                              tested_python_requires=tested_python_requires)

        # Export does a check on existing name & version
        if name:
            if conanfile.name and name != conanfile.name:
                raise ConanException("Package recipe with name %s!=%s" % (name, conanfile.name))
            conanfile.name = name

        if version:
            if conanfile.version and version != conanfile.version:
                raise ConanException("Package recipe with version %s!=%s"
                                     % (version, conanfile.version))
            conanfile.version = version

        if user:
            if conanfile.user and user != conanfile.user:
                raise ConanException("Package recipe with user %s!=%s"
                                     % (user, conanfile.user))
            conanfile.user = user

        if channel:
            if conanfile.channel and channel != conanfile.channel:
                raise ConanException("Package recipe with channel %s!=%s"
                                     % (channel, conanfile.channel))
            conanfile.channel = channel

        if conanfile.channel and not conanfile.user:
            raise ConanException(f"{conanfile_path}: Can't specify channel '{conanfile.channel}' without user")

        if hasattr(conanfile, "set_name"):
            with conanfile_exception_formatter("conanfile.py", "set_name"):
                conanfile.set_name()
        if hasattr(conanfile, "set_version"):
            with conanfile_exception_formatter("conanfile.py", "set_version"):
                conanfile.set_version()

        return conanfile

    def load_export(self, conanfile_path, name, version, user, channel, graph_lock=None,
                    remotes=None):
        """ loads the conanfile and evaluates its name, version, and enforce its existence
        """
        conanfile = self.load_named(conanfile_path, name, version, user, channel, graph_lock,
                                    remotes=remotes)
        if not conanfile.name:
            raise ConanException("conanfile didn't specify name")
        if not conanfile.version:
            raise ConanException("conanfile didn't specify version")

        ref = RecipeReference(conanfile.name, conanfile.version, conanfile.user, conanfile.channel)
        conanfile.display_name = str(ref)
        conanfile.output.scope = conanfile.display_name
        return conanfile

    def load_consumer(self, conanfile_path, name=None, version=None, user=None,
                      channel=None, graph_lock=None,  remotes=None, update=None, check_update=None,
                      tested_python_requires=None):
        """ loads a conanfile.py in user space. Might have name/version or not
        """
        conanfile = self.load_named(conanfile_path, name, version, user, channel, graph_lock,
                                    remotes, update, check_update,
                                    tested_python_requires=tested_python_requires)

        ref = RecipeReference(conanfile.name, conanfile.version, conanfile.user, conanfile.channel)
        if str(ref):
            conanfile.display_name = "%s (%s)" % (os.path.basename(conanfile_path), str(ref))
        else:
            conanfile.display_name = os.path.basename(conanfile_path)
        conanfile.output.scope = conanfile.display_name
        conanfile._conan_is_consumer = True
        return conanfile

    def load_conanfile(self, conanfile_path, ref, graph_lock=None, remotes=None,
                       update=None, check_update=None):
        """ load a conanfile with a full reference, name, version, user and channel are obtained
        from the reference, not evaluated. Main way to load from the cache
        """
        try:
            conanfile, _ = self.load_basic_module(conanfile_path, graph_lock, str(ref), remotes,
                                                  update=update, check_update=check_update)
        except Exception as e:
            raise ConanException("%s: Cannot load recipe.\n%s" % (str(ref), str(e)))

        conanfile.name = ref.name
        conanfile.version = str(ref.version)
        conanfile.user = ref.user
        conanfile.channel = ref.channel
        return conanfile

    def load_conanfile_txt(self, conan_txt_path):
        if not os.path.exists(conan_txt_path):
            raise NotFoundException("Conanfile not found!")

        try:
            contents = load_user_encoded(conan_txt_path)
        except Exception as e:
            raise ConanException(f"Cannot load conanfile.txt:\n{e}")
        path, basename = os.path.split(conan_txt_path)
        display_name = basename
        conanfile = self._parse_conan_txt(contents, path, display_name)
        conanfile._conan_helpers = self._conanfile_helpers
        conanfile._conan_is_consumer = True
        return conanfile

    @staticmethod
    def _parse_conan_txt(contents, path, display_name):
        conanfile = ConanFile(display_name)

        try:
            parser = ConanFileTextLoader(contents)
        except Exception as e:
            raise ConanException("%s:\n%s" % (path, str(e)))
        for reference in parser.requirements:
            conanfile.requires(reference)
        for build_reference in parser.tool_requirements:
            # TODO: Improve this interface
            conanfile.requires.tool_require(build_reference)
        for ref in parser.test_requirements:
            # TODO: Improve this interface
            conanfile.requires.test_require(ref)

        if parser.layout:
            layout_method = {"cmake_layout": cmake_layout,
                             "vs_layout": vs_layout,
                             "bazel_layout": bazel_layout}.get(parser.layout)
            if not layout_method:
                raise ConanException("Unknown predefined layout '{}' declared in "
                                     "conanfile.txt".format(parser.layout))

            def layout(_self):
                layout_method(_self)

            conanfile.layout = types.MethodType(layout, conanfile)

        conanfile.generators = parser.generators
        try:
            conanfile.options = Options.loads(parser.options)
        except Exception:
            raise ConanException("Error while parsing [options] in conanfile.txt\n"
                                 "Options should be specified as 'pkg/*:option=value'")
        return conanfile

    def load_virtual(self, requires=None, tool_requires=None, python_requires=None, graph_lock=None,
                     remotes=None, update=None, check_updates=None):
        # If user don't specify namespace in options, assume that it is
        # for the reference (keep compatibility)
        conanfile = ConanFile(display_name="cli")
        conanfile._conan_helpers = self._conanfile_helpers

        if tool_requires:
            for reference in tool_requires:
                conanfile.requires.tool_require(repr(reference))
        if requires:
            for reference in requires:
                conanfile.requires(repr(reference))

        if python_requires:
            conanfile.python_requires = [pr.repr_notime() for pr in python_requires]

        if self._pyreq_loader:
            self._pyreq_loader.load_py_requires(conanfile, self, graph_lock, remotes,
                                                update, check_updates)

        conanfile._conan_is_consumer = True
        conanfile.generators = []  # remove the default txt generator
        return conanfile


def _parse_module(conanfile_module, module_id):
    """ Parses a python in-memory module, to extract the classes, mainly the main
    class defining the Recipe, but also process possible existing generators
    @param conanfile_module: the module to be processed
    @return: the main ConanFile class from the module
    """
    result = None
    for name, attr in conanfile_module.__dict__.items():
        if (name.startswith("_") or not inspect.isclass(attr) or
                attr.__dict__.get("__module__") != module_id):
            continue

        if issubclass(attr, ConanFile) and attr != ConanFile:
            if result is None:
                result = attr
            else:
                raise ConanException("More than 1 conanfile in the file")

    if result is None:
        raise ConanException("No subclass of ConanFile")

    return result


_load_python_lock = Lock()  # Loading our Python files is not thread-safe (modifies sys)


def _parse_conanfile(conanfile_path):
    with _load_python_lock:
        module, module_id = _load_python_file(conanfile_path)
    try:
        conanfile = _parse_module(module, module_id)
        return module, conanfile
    except Exception as e:  # re-raise with file name
        raise ConanException("%s: %s" % (conanfile_path, str(e)))


def load_python_file(conan_file_path):
    """ From a given path, obtain the in memory python import module
    """
    with _load_python_lock:
        module, module_id = _load_python_file(conan_file_path)
    return module, module_id


def _load_python_file(conan_file_path):
    """ From a given path, obtain the in memory python import module
    """

    if not os.path.exists(conan_file_path):
        raise NotFoundException("%s not found!" % conan_file_path)

    def new_print(*args, **kwargs):  # Make sure that all user python files print() goes to stderr
        kwargs.setdefault("file", sys.stderr)
        print(*args, **kwargs)

    module_id = str(uuid.uuid1())
    current_dir = os.path.dirname(conan_file_path)
    sys.path.insert(0, current_dir)
    try:
        old_modules = list(sys.modules.keys())
        with chdir(current_dir):
            old_dont_write_bytecode = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                spec = imp_util.spec_from_file_location(module_id, conan_file_path)
                loaded = imp_util.module_from_spec(spec)
                spec.loader.exec_module(loaded)
                sys.dont_write_bytecode = old_dont_write_bytecode
            except ImportError:
                version_txt = _get_required_conan_version_without_loading(conan_file_path)
                if version_txt:
                    validate_conan_version(version_txt)
                raise

            required_conan_version = getattr(loaded, "required_conan_version", None)
            if required_conan_version:
                validate_conan_version(required_conan_version)

        # These lines are necessary, otherwise local conanfile imports with same name
        # collide, but no error, and overwrite other packages imports!!
        added_modules = set(sys.modules).difference(old_modules)
        for added in added_modules:
            module = sys.modules[added]
            if module:
                try:
                    try:
                        # Most modules will have __file__ != None
                        folder = os.path.dirname(module.__file__)
                    except (AttributeError, TypeError):
                        # But __file__ might not exist or equal None
                        # Like some builtins and Namespace packages py3
                        folder = module.__path__._path[0]
                except AttributeError:  # In case the module.__path__ doesn't exist
                    pass
                else:
                    if folder.startswith(current_dir):
                        module = sys.modules.pop(added)
                        module.print = new_print
                        sys.modules["%s.%s" % (module_id, added)] = module
    except ConanException:
        raise
    except Exception:
        trace = traceback.format_exc().split('\n')
        raise ConanException("Unable to load conanfile in %s\n%s" % (conan_file_path,
                                                                     '\n'.join(trace[3:])))
    finally:
        sys.path.pop(0)

    loaded.print = new_print
    return loaded, module_id


def _get_required_conan_version_without_loading(conan_file_path):
    # First, try to detect the required_conan_version in "text" mode
    # https://github.com/conan-io/conan/issues/11239
    contents = load(conan_file_path)

    txt_version = None

    try:
        found = re.search(r"(.*)required_conan_version\s*=\s*[\"'](.*)[\"']", contents)
        if found and "#" not in found.group(1):
            txt_version = found.group(2)
    except:  # noqa this should be solid, cannot fail
        pass

    return txt_version


class ConanFileTextLoader:
    """Parse a conanfile.txt file"""

    def __init__(self, input_text):
        # Prefer composition over inheritance, the __getattr__ was breaking things
        self._config_parser = TextINIParse(input_text,  ["requires", "generators", "options",
                                                         "imports", "tool_requires", "test_requires",
                                                         "layout"],
                                           strip_comments=True)

    @property
    def layout(self):
        """returns the declared layout"""
        tmp = [r.strip() for r in self._config_parser.layout.splitlines()]
        if len(tmp) > 1:
            raise ConanException("Only one layout can be declared in the [layout] section of "
                                 "the conanfile.txt")
        return tmp[0] if tmp else None

    @property
    def requirements(self):
        """returns a list of requires
        EX:  "OpenCV/2.4.10@phil/stable"
        """
        return [r.strip() for r in self._config_parser.requires.splitlines()]

    @property
    def tool_requirements(self):
        """returns a list of tool_requires
        EX:  "OpenCV/2.4.10@phil/stable"
        """

        return [r.strip() for r in self._config_parser.tool_requires.splitlines()]

    @property
    def test_requirements(self):
        """returns a list of test_requires
        EX:  "gtest/2.4.10@phil/stable"
        """

        return [r.strip() for r in self._config_parser.test_requires.splitlines()]

    @property
    def options(self):
        return self._config_parser.options

    @property
    def generators(self):
        return self._config_parser.generators.splitlines()
