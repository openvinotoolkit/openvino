from jinja2 import Template
from os import path, remove
from shutil import rmtree


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


def load_secret(path: str = "../.secret") -> str:
    """Loading secret file

    :param path: Path to secret file. Defaults to "../.secret".
    :type path: str
    :returns: Secret key
    :rtype: str

    """
    with open(path, "r+") as file:
        return file.readline().strip()


def process_notebook_name(notebook_name: str) -> str:
    """Processes notebook name

    :param notebook_name: Notebook name by default keeps convention:
        [3 digit]-name-with-dashes-with-output.rst,
        example: 001-hello-world-with-output.rst
    :type notebook_name: str
    :returns: Processed notebook name,
        001-hello-world-with-output.rst -> 001. hello world
    :rtype: str

    """
    return (
        notebook_name[:3]
        + "."
        + " ".join(notebook_name[4:].split(".")[0].split("-")[:-2])
    )


def find_latest_artifact(artifacts_dict: dict, name: str = "rst_files") -> int:
    """Finds id of latest artifact that can be downloaded

    :param artifacts_dict: Fetched github actions
    :type artifacts_dict: dict
    :param name: Name of searched artifact. Defaults to "rst_files".
    :type name: str
    :returns: Id of latest artifact containing rst files
    :rtype: int

    """
    return max([r["id"] for r in artifacts_dict["artifacts"] if r["name"] == name])


def verify_notebook_name(notebook_name: str) -> bool:
    """Verification based on notebook name

    :param notebook_name: Notebook name by default keeps convention:
        [3 digit]-name-with-dashes-with-output.rst,
        example: 001-hello-world-with-output.rst
    :type notebook_name: str
    :returns: Return if notebook meets requirements
    :rtype: bool

    """
    return notebook_name[:3].isdigit() and notebook_name[-4:] == ".rst"


def generate_artifact_link(owner: str, name: str) -> str:
    """Generate link for downloading artifacts

    :param owner: Github repo owner name
    :type owner: str
    :param name: Github repo name
    :type name: str
    :returns: Link to api to download artifacts
    :rtype: str

    """
    return f"https://api.github.com/repos/{owner}/{name}/actions/artifacts"


def remove_existing(notebooks_path: str) -> None:
    """Removes file if already existed

    :param notebooks_path: path to file to be removed
    :type notebooks_path: str

    """
    if path.exists(notebooks_path):
        if path.isdir(notebooks_path):
            rmtree(notebooks_path)
        else:
            remove(notebooks_path)

def split_notebooks_into_sections(notebooks: list) -> list:
    series = [list() for _ in range(5)]
    for notebook in notebooks:
        try:
            series[int(notebook.name[0])].append(notebook)
        except IndexError:
            pass
    return series