import textwrap

import jinja2
from jinja2 import Template


class TargetsTemplate2:
    """
    FooTargets.cmake
    """
    def __init__(self, cmakedeps, conanfile):
        self._cmakedeps = cmakedeps
        self._conanfile = conanfile

    def content(self):
        t = Template(self._template, trim_blocks=True, lstrip_blocks=True,
                     undefined=jinja2.StrictUndefined)
        return t.render(self._context)

    @property
    def filename(self):
        f = self._cmakedeps.get_cmake_filename(self._conanfile)
        return f"{f}Targets.cmake"

    @property
    def _context(self):
        filename = self._cmakedeps.get_cmake_filename(self._conanfile)
        ret = {"ref": str(self._conanfile.ref),
               "filename": filename}
        return ret

    @property
    def _template(self):
        return textwrap.dedent("""\
            include_guard()
            message(STATUS "Conan: Configuring Targets for {{ ref }}")

            # Load information for each installed configuration.
            file(GLOB _target_files "${CMAKE_CURRENT_LIST_DIR}/{{filename}}-Targets-*.cmake")
            foreach(_target_file IN LISTS _target_files)
              include("${_target_file}")
            endforeach()

            file(GLOB _build_files "${CMAKE_CURRENT_LIST_DIR}/{{filename}}-TargetsBuild-*.cmake")
            foreach(_build_file IN LISTS _build_files)
              include("${_build_file}")
            endforeach()
            """)
