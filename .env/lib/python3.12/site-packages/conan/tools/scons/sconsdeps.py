from jinja2 import Template

from conan.tools import CppInfo
from conan.internal.util.files import save


class SConsDeps:
    def __init__(self, conanfile):
        self._conanfile = conanfile
        self._ordered_deps = None
        self._generator_file = 'SConscript_conandeps'

    @property
    def ordered_deps(self):
        if self._ordered_deps is None:
            deps = self._conanfile.dependencies.host.topological_sort
            self._ordered_deps = [dep for dep in reversed(deps.values())]
        return self._ordered_deps

    def _get_cpp_info(self):
        ret = CppInfo(self._conanfile)
        for dep in self.ordered_deps:
            dep_cppinfo = dep.cpp_info.aggregated_components()
            ret.merge(dep_cppinfo)
        return ret

    def generate(self):
        save(self._generator_file, self._content)

    @property
    def _content(self):
        template = Template("""
        "{{dep_name}}" : {
            "CPPPATH"     : {{info.includedirs or []}},
            "LIBPATH"     : {{info.libdirs or []}},
            "BINPATH"     : {{info.bindirs or []}},
            "LIBS"        : {{(info.libs or []) + (info.system_libs or [])}},
            "FRAMEWORKS"  : {{info.frameworks or []}},
            "FRAMEWORKPATH" : {{info.frameworkdirs or []}},
            "CPPDEFINES"  : {{info.defines or []}},
            "CXXFLAGS"    : {{info.cxxflags or []}},
            "CCFLAGS"     : {{info.cflags or []}},
            "SHLINKFLAGS" : {{info.sharedlinkflags or []}},
            "LINKFLAGS"   : {{info.exelinkflags or []}},
        },
        {% if dep_version is not none %}"{{dep_name}}_version" : "{{dep_version}}",{% endif %}
        """)
        sections = ["conandeps = {\n"]
        all_flags = template.render(dep_name="conandeps", dep_version=None,
                                    info=self._get_cpp_info())
        sections.append(all_flags)

        # TODO: Add here in 2.0 the "skip": False trait
        host_req = self._conanfile.dependencies.filter({"build": False}).values()
        for dep in host_req:
            dep_flags = template.render(dep_name=dep.ref.name, dep_version=dep.ref.version,
                                        info=dep.cpp_info)
            sections.append(dep_flags)
        sections.append("}\n")
        sections.append("Return('conandeps')\n")
        return "\n".join(sections)
