import os

from conan.internal.model.options import Options
from conan.internal.model.profile import Profile
from conan.internal.util.files import save


def create_profile(folder, name, settings=None, package_settings=None, options=None):
    profile = Profile()
    profile.settings = settings or {}

    if package_settings:
        profile.package_settings = package_settings

    if options:
        profile.options = Options(options_values=options)

    save(os.path.join(folder, name), profile.dumps())
