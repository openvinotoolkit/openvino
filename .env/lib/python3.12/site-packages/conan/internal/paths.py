import os
import platform
from pathlib import Path

from conan.errors import ConanException

if platform.system() == "Windows":
    def _conan_expand_user(path):
        """ wrapper to the original expanduser function, to workaround python returning
        verbatim %USERPROFILE% when some other app (git for windows) sets HOME envvar
        """
        path = str(path)
        if path[0] != '~':
            return path
        # In win these variables should exist and point to user directory, which
        # must exist.
        home = os.environ.get("HOME")
        try:
            # Problematic cases of wrong HOME variable
            # - HOME = %USERPROFILE% verbatim, as messed by some other tools
            # - MSYS console, that defines a different user home in /c/mingw/msys/users/xxx
            # In these cases, it is safe to remove it and rely on USERPROFILE directly
            if home and (not os.path.exists(home) or
                         (os.getenv("MSYSTEM") and os.getenv("USERPROFILE"))):
                del os.environ["HOME"]
            result = os.path.expanduser(path)
        finally:
            if home is not None:
                os.environ["HOME"] = home
        return result
else:
    _conan_expand_user = os.path.expanduser

DEFAULT_CONAN_HOME = ".conan2"


def find_file_walk_up(start, filename, end=None):
    path = Path(start)
    end = Path(end) if end else None
    while True:
        file = path / filename
        if file.is_file():
            return file
        if len(path.parts) == 1:  # finish at '/'
            break
        if end and path == end:
            break
        path = path.parent
    return None


def get_conan_user_home():

    def _user_home_from_conanrc_file():
        try:
            conanrc_path = find_file_walk_up(os.getcwd(), ".conanrc")

            with open(conanrc_path) as conanrc_file:
                values = {k: str(v) for k, v in
                          (line.split('=') for line in conanrc_file.read().splitlines() if
                           not line.startswith("#"))}

            conan_home = values["conan_home"]
            # check if it's a local folder
            if conan_home[:2] in ("./", ".\\") or conan_home.startswith(".."):
                conan_home = conanrc_path.parent.absolute() / conan_home
            return conan_home
        except (OSError, KeyError, TypeError):
            return None

    user_home = _user_home_from_conanrc_file() or os.getenv("CONAN_HOME")
    if user_home is None:
        # the default, in the user home
        user_home = os.path.join(_conan_expand_user("~"), DEFAULT_CONAN_HOME)
    else:  # Do an expansion, just in case the user is using ~/something/here
        user_home = _conan_expand_user(user_home)
    if not os.path.isabs(user_home):
        raise ConanException("Invalid CONAN_HOME value '%s', "
                             "please specify an absolute or path starting with ~/ "
                             "(relative to user home)" % user_home)
    return user_home


# Files
CONANFILE = 'conanfile.py'
CONANFILE_TXT = "conanfile.txt"
CONAN_MANIFEST = "conanmanifest.txt"
CONANINFO = "conaninfo.txt"
PACKAGE_TGZ_NAME = "conan_package.tgz"
EXPORT_TGZ_NAME = "conan_export.tgz"
EXPORT_SOURCES_TGZ_NAME = "conan_sources.tgz"
DATA_YML = "conandata.yml"
