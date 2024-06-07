from jinja2 import Template


def create_content(template: str, notebooks_data: dict, file_name: str):
    """Filling rst template with data

    :param template: jinja template that will be filled with notebook data
    :type template: str
    :param notebooks_data: data structure containing information required to fill template
    :type notebooks_data: dict
    :param file_name: file name
    :type file_name: str
    :returns: Filled template
    :rtype: str

    """
    template = Template(template)
    notebooks_data["notebook"] = "-".join(file_name.split("-")[:-2])
    return template.render(notebooks_data)


def add_content_below(text: str, path: str, line=3) -> bool:
    """Add additional content (like binder button) to existing rst file

    :param text: Text that will be added inside rst file
    :type text: str
    :param path: Path to modified file
    :type path: str
    :param line: Line number that content will be added. Defaults to 3.
    :type line: int
    :returns: Informs about success or failure in modifying file
    :rtype: bool

    """
    try:
        with open(path, "r+", encoding="utf-8") as file:
            current_file = file.readlines()
            current_file[line:line] = text
            file.seek(0)
            file.writelines(current_file)
            return True
    except FileNotFoundError:
        return False


def verify_notebook_name(notebook_name: str) -> bool:
    """Verification based on notebook name

    :param notebook_name: Notebook name by default keeps convention:
        name-with-dashes-with-output.rst,
        example: hello-world-with-output.rst
    :type notebook_name: str
    :returns: Return if notebook meets requirements
    :rtype: bool

    """
    return notebook_name[-16:] == "-with-output.rst"