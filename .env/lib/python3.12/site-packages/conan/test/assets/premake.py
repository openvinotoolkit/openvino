import textwrap

from jinja2 import Template


def gen_premake5(workspace, projects, includedirs=None, configurations=None):
    includedirs = includedirs if includedirs is not None else ["."]
    premake5 = textwrap.dedent("""\
        workspace "{{workspace}}"
            cppdialect "C++17"
            configurations {{premake_quote(configurations)}}
            fatalwarnings {"All"}
            floatingpoint "Fast"
            includedirs {{premake_quote(includedirs)}}
            filter "configurations:Debug"
               defines { "DEBUG" }
               symbols "On"

            filter "configurations:Release"
               defines { "NDEBUG" }
               optimize "On"

        {% for project in projects %}
        project "{{project["name"]}}"
            kind "{{project["kind"] or 'StaticLib'}}"
            language "{{project["language"] or 'C++'}}"
            files {{premake_quote(project["files"])}}
            links {{premake_quote(project["links"])}}
        {% endfor %}
        """)

    t = Template(premake5, trim_blocks=True, lstrip_blocks=True)

    def premake_quote(s):
        return '{' + ', '.join(['"{}"'.format(s) for s in s]) + '}'

    return t.render(
        {
            "premake_quote": premake_quote,
            "workspace": workspace,
            "configurations": configurations or ["Debug", "Release"],
            "includedirs": includedirs,
            "projects": projects,
        }
    )
