from collections import defaultdict
from io import StringIO

from conan import ConanFile
from conan.errors import ConanException
from conan.internal.conan_app import ConanFileHelpers
from conan.internal.model.conf import Conf
from conan.internal.model.layout import Folders, Infos
from conan.internal.model.options import Options


class RedirectedInputStream:
    """
    Mock for testing. If get_username or get_password is requested will raise
    an exception except we have a value to return.
    """

    def __init__(self, answers: list):
        self.answers = answers

    def readline(self):
        if not self.answers:
            raise Exception("\n\n**********\n\nClass MockedInputStream: "
                            "There are no more inputs to be returned.\n"
                            "CHECK THE 'inputs=[]' ARGUMENT OF THE TESTCLIENT\n**********+*\n\n\n")
        ret = self.answers.pop(0)
        return ret


class MockSettings(object):

    def __init__(self, values):
        self.values = values

    def get_safe(self, value, default=None):
        return self.values.get(value, default)

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise ConanException("'%s' value not defined" % name)

    def rm_safe(self, name):
        self.values.pop(name, None)

    def possible_values(self):
        return defaultdict(lambda: [])


class ConanFileMock(ConanFile):
    def __init__(self, settings=None, options=None, runner=None, display_name=""):
        self.display_name = display_name
        self._conan_node = None
        self.package_type = "unknown"
        self.settings = settings or MockSettings({"os": "Linux", "arch": "x86_64"})
        self.settings_build = settings or MockSettings({"os": "Linux", "arch": "x86_64"})
        self.settings_target = None
        self.runner = runner
        self.options = options or Options()
        self.generators = []
        self.conf = Conf()
        self.conf_build = Conf()
        self.folders = Folders()
        self.folders.set_base_source(".")
        self.folders.set_base_export_sources(".")
        self.folders.set_base_build(".")
        self.folders.set_base_generators(".")
        self.cpp = Infos()
        self.env_scripts = {}
        self.system_requires = {}
        self.win_bash = None
        self.command = None
        self._commands = []
        self._conan_helpers = ConanFileHelpers(None, None, self.conf, None, None)

    def run(self, *args, **kwargs):
        self.command = args[0]
        self._commands.append(args[0])
        if self.runner:
            kwargs.pop("quiet", None)
            return self.runner(*args, **kwargs)
        return 0  # simulating it was OK!

    @property
    def commands(self):
        result = self._commands
        self._commands = []
        return result


MockOptions = MockSettings


class RedirectedTestOutput(StringIO):
    def __init__(self):
        # Chage to super() for Py3
        StringIO.__init__(self)

    def clear(self):
        self.seek(0)
        self.truncate(0)

    def __repr__(self):
        return self.getvalue()

    def __str__(self, *args, **kwargs):
        return self.__repr__()

    def __eq__(self, value):
        return self.__repr__() == value

    def __contains__(self, value):
        return value in self.__repr__()
