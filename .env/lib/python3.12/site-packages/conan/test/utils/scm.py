import os

from conan.test.utils.test_files import temp_folder
from conan.internal.util.files import save_files, chdir
from conan.internal.util.runners import detect_runner


def git_create_bare_repo(folder=None, reponame="repo.git"):
    folder = folder or temp_folder()
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        detect_runner('git init --bare {}'.format(reponame))
        return os.path.join(folder, reponame).replace("\\", "/")
    finally:
        os.chdir(cwd)


def create_local_git_repo(files=None, branch=None, submodules=None, folder=None, commits=1,
                          tags=None, origin_url=None, main_branch="master"):
    tmp = folder or temp_folder()
    if files:
        save_files(tmp, files)

    def _run(cmd, p):
        with chdir(p):
            _, out = detect_runner("git {}".format(cmd))
            return out.strip()

    _run("init .", tmp)
    _run('config user.name "Your Name"', tmp)
    _run('config user.email "you@example.com"', tmp)
    _run("checkout -b {}".format(branch or main_branch), tmp)

    _run("add .", tmp)
    for i in range(0, commits):
        _run('commit --allow-empty -m "commiting"', tmp)

    tags = tags or []
    for tag in tags:
        _run("tag %s" % tag, tmp)

    if submodules:
        for submodule in submodules:
            _run('submodule add "%s"' % submodule, tmp)
        _run('commit -m "add submodules"', tmp)

    if origin_url:
        _run('remote add origin {}'.format(origin_url), tmp)

    commit = _run('rev-list HEAD -n 1', tmp)
    return tmp.replace("\\", "/"), commit


def git_add_changes_commit(folder, msg="fix"):
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        # Make sure user and email exist, otherwise it can error
        detect_runner('git config user.name "Your Name"')
        detect_runner('git config user.email "you@example.com"')
        detect_runner('git add . && git commit -m "{}"'.format(msg))
        _, out = detect_runner("git rev-parse HEAD")
        return out.strip()
    finally:
        os.chdir(cwd)
