from conan.errors import ConanException


def is_multi_configuration(generator):
    if not generator:
        return False
    return "Visual" in generator or "Xcode" in generator or "Multi-Config" in generator


def parse_extra_variable(source, key, value):
    CMAKE_CACHE_TYPES = ["BOOL", "FILEPATH", "PATH", "STRING", "INTERNAL"]
    if isinstance(value, str):
        return f"\"{value}\""
    elif isinstance(value, (int, float)):
        return value
    elif isinstance(value, dict):
        var_value = parse_extra_variable(source, key, value.get("value"))
        is_force = value.get("force")
        if is_force:
            if not isinstance(is_force, bool):
                raise ConanException(f'{source} "{key}" "force" must be a boolean')
        is_cache = value.get("cache")
        if is_cache:
            if not isinstance(is_cache, bool):
                raise ConanException(f'{source} "{key}" "cache" must be a boolean')
            var_type = value.get("type")
            if not var_type:
                raise ConanException(f'{source} "{key}" needs "type" defined for cache variable')
            if var_type not in CMAKE_CACHE_TYPES:
                raise ConanException(f'{source} "{key}" invalid type "{var_type}" for cache variable.'
                                     f' Possible types: {", ".join(CMAKE_CACHE_TYPES)}')
            # Set docstring as variable name if not defined
            docstring = value.get("docstring") or key
            force_str = " FORCE" if is_force else ""  # Support python < 3.11
            return f"{var_value} CACHE {var_type} \"{docstring}\"{force_str}"
        else:
            if is_force:
                raise ConanException(f'{source} "{key}" "force" is only allowed for cache variables')
            return var_value
    raise ConanException(f'{source} "{key}" has invalid type. Allowed types: str, int, float, dict,'
                         f' got {type(value)}')


def cmake_escape_value(v):
    return v.replace('\\', '\\\\').replace('$', '\\$').replace('"', '\\"')
