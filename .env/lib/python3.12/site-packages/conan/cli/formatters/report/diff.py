import json
import os
import base64

from jinja2 import Template

from conan.api.output import cli_out_write
from conan.cli.formatters.report.diff_html import diff_html


def _generate_json(result):
    diff_text = result["diff"]
    src_prefix = result["src_prefix"]
    dst_prefix = result["dst_prefix"]
    ret = {}
    current_filename = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            src_filename, dst_filename = _get_filenames(line, src_prefix, dst_prefix)
            current_filename = src_filename
            ret[current_filename] = [line]
        else:
            ret[current_filename].append(line)
    return ret


def _get_filenames(line, src_prefix, dst_prefix):
    """
    Extracts the source and destination filenames from a diff line.
    """
    src_index = line.find(src_prefix)
    dst_index = line.find(dst_prefix)

    if src_index == -1 or dst_index == -1:
        return None, None

    src_filename = line[src_index + len(src_prefix) - 1:dst_index - 1].strip()
    dst_filename = line[dst_index + len(dst_prefix) - 1:].strip()

    return src_filename, dst_filename


def _render_diff(content, template, template_folder, **kwargs):
    from conan import __version__
    template = Template(template, autoescape=True)

    def _safe_filename(filename):
        # Calculate base64 of the filename
        return base64.b64encode(filename.encode(), altchars=b'-_').decode()

    def _remove_prefixes(line):
        return line.replace(kwargs["src_prefix"][:-1], "").replace(kwargs["dst_prefix"][:-1], "")

    def _replace_cache_paths(line):
        return line.replace(kwargs["old_cache_path"], "(old)").replace(kwargs["new_cache_path"], "(new)")

    def _replace_paths(line):
        return _remove_prefixes(_replace_cache_paths(line))

    def _extract_header(diff_lines):
        # Header ends at the first occurrence of +++ line,
        # and it can be at most 10 lines long
        for i, line in enumerate(diff_lines[:10]):
            if line.startswith("+++ "):
                return diff_lines[:i + 1]
        return diff_lines[:10]

    def _parse_header_is_deleted(header_contents):
        return ("+++ /dev/null" in header_contents
                or any("deleted file mode" in line for line in header_contents))

    def _parse_header_rename_to(header_contents):
        if not any("similarity index" in line for line in header_contents):
            return None
        for line in header_contents:
            if line.startswith("rename to "):
                return line[len("rename to "):]
        return None

    per_folder = {"folders": {}, "files": {}}
    for file in content:
        header = _extract_header(content[file])
        renamed_to = _parse_header_rename_to(header)
        replaced_path = _replace_paths(renamed_to or file)
        replaced_file = replaced_path.replace("(old)", "").replace("(new)", "").replace("\\", "/")
        bits = replaced_file.split("/")[1:]
        cur = per_folder
        for folder in bits[:-1]:
            cur = cur["folders"].setdefault(folder, {"folders": {}, "files": {}})
        filename = bits[-1]
        cur["files"][filename] = {"filename": file,  # This is file so renamed use old name
                                  "is_new": "(new)" in replaced_path,
                                  "is_deleted": _parse_header_is_deleted(header),
                                  "renamed_to": renamed_to,
                                  "relative_path": replaced_path}

    def flatten_empty_folders(current_node):
        for folder_data in current_node["folders"].values():
            flatten_empty_folders(folder_data)

        promoted_folders = {}

        # The list here is important to avoid modifying the dict while iterating
        for folder_name, folder_data in list(current_node["folders"].items()):
            if not folder_data["files"]:
                for sub_folder_name, sub_folder_data in folder_data['folders'].items():
                    new_key = os.path.join(folder_name, sub_folder_name)
                    promoted_folders[new_key] = sub_folder_data

                del current_node["folders"][folder_name]

        current_node["folders"].update(promoted_folders)

    flatten_empty_folders(per_folder)

    # Now sort each folder and file recursively
    def sort_folders_and_files(node):
        node["folders"] = dict(sorted(node["folders"].items()))
        node["files"] = dict(sorted(node["files"].items(), key=lambda x: x[0].lower()))
        for folder_data in node["folders"].values():
            sort_folders_and_files(folder_data)
    sort_folders_and_files(per_folder)

    return template.render(content=content,
                           per_folder=per_folder,
                           base_template_path=template_folder, version=__version__,
                           safe_filename=_safe_filename,
                           replace_paths=_replace_paths,
                           replace_cache_paths=_replace_cache_paths,
                           remove_prefixes=_remove_prefixes,
                           **kwargs)


def format_diff_html(result):
    conan_api = result["conan_api"]

    template_folder = os.path.join(conan_api.cache_folder, "templates")
    user_template = os.path.join(template_folder, "diff.html")
    template = diff_html
    if os.path.isfile(user_template):
        with open(user_template, 'r', encoding="utf-8", newline="") as handle:
            template = handle.read()

    content = _generate_json(result)

    cli_out_write(_render_diff(content, template, template_folder,
                               old_reference=result["old_export_ref"],
                               new_reference=result["new_export_ref"],
                               old_cache_path=result["old_cache_path"],
                               new_cache_path=result["new_cache_path"],
                               src_prefix=result["src_prefix"],
                               dst_prefix=result["dst_prefix"]))


def format_diff_txt(result):
    diff_text = result["diff"]
    cli_out_write(diff_text)


def format_diff_json(result):
    cli_out_write(json.dumps(_generate_json(result), indent=2))
