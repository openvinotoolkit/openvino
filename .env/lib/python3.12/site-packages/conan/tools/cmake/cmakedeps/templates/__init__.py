import jinja2
from jinja2 import Template

from conan.errors import ConanException


class CMakeDepsFileTemplate(object):

    def __init__(self, cmakedeps, require, conanfile, generating_module=False):
        self.cmakedeps = cmakedeps
        self.require = require
        self.conanfile = conanfile
        self.generating_module = generating_module

    @property
    def pkg_name(self):
        return self.conanfile.ref.name + self.suffix

    @property
    def root_target_name(self):
        return self.get_root_target_name(self.conanfile, self.suffix)

    @property
    def file_name(self):
        return self.cmakedeps.get_cmake_package_name(self.conanfile, module_mode=self.generating_module) + self.suffix

    @property
    def suffix(self):
        if not self.require.build:
            return ""
        return self.cmakedeps.build_context_suffix.get(self.conanfile.ref.name, "")

    def render(self):
        try:
            context = self.context
        except Exception as e:
            raise ConanException("error generating context for '{}': {}".format(self.conanfile, e))

        # Cache the template instance as a class attribute to greatly speed up the rendering
        # NOTE: this assumes that self.template always returns the same string
        template_instance = getattr(type(self), "template_instance", None)
        if template_instance is None:
            template_instance = Template(self.template, trim_blocks=True, lstrip_blocks=True,
                                         undefined=jinja2.StrictUndefined)
            setattr(type(self), "template_instance", template_instance)
        return template_instance.render(context)

    def context(self):
        raise NotImplementedError()

    @property
    def template(self):
        raise NotImplementedError()

    @property
    def filename(self):
        raise NotImplementedError()

    @property
    def configuration(self):
        return self.cmakedeps.configuration

    @property
    def arch(self):
        return self.cmakedeps.arch

    @property
    def config_suffix(self):
        return "_{}".format(self.configuration.upper()) if self.configuration else ""

    @staticmethod
    def _get_target_default_name(req, component_name="", suffix=""):
        return "{name}{suffix}::{cname}{suffix}".format(cname=component_name or req.ref.name,
                                                        name=req.ref.name, suffix=suffix)

    def get_root_target_name(self, req, suffix=""):
        if self.generating_module:
            ret = self.cmakedeps.get_property("cmake_module_target_name", req)
            if ret:
                return ret
        ret = self.cmakedeps.get_property("cmake_target_name", req)
        return ret or self._get_target_default_name(req, suffix=suffix)

    def get_component_alias(self, req, comp_name):
        if comp_name not in req.cpp_info.components:
            # foo::foo might be referencing the root cppinfo
            if req.ref.name == comp_name:
                return self.get_root_target_name(req)
            raise ConanException("Component '{name}::{cname}' not found in '{name}' "
                                 "package requirement".format(name=req.ref.name, cname=comp_name))
        if self.generating_module:
            ret = self.cmakedeps.get_property("cmake_module_target_name", req, comp_name=comp_name)
            if ret:
                return ret
        ret = self.cmakedeps.get_property("cmake_target_name", req, comp_name=comp_name)

        # If we don't specify the `cmake_target_name` property for the component it will
        # fallback to the pkg_name::comp_name, it wont use the root cpp_info cmake_target_name
        # property because that is also an absolute name (Greetings::Greetings), it is not a namespace
        # and we don't want to split and do tricks.
        return ret or self._get_target_default_name(req, component_name=comp_name)
