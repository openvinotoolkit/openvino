import fnmatch
import os

from conan.api.output import Color
from conan.tools.files import chdir, update_conandata
from conan.errors import ConanException
from conan.internal.model.conf import ConfDefinition
from conan.internal.util.files import mkdir
from conan.internal.util.runners import check_output_runner


class Git:
    """
    Git is a wrapper for several common patterns used with *git* tool.
    """
    def __init__(self, conanfile, folder=".", excluded=None):
        """
        :param conanfile: Conanfile instance.
        :param folder: Current directory, by default ``.``, the current working directory.
        :param excluded: Files to be excluded from the "dirty" checks. It will compose with the
          configuration ``core.scm:excluded`` (the configuration has higher priority).
          It is a list of patterns to ``fnmatch``.
        """
        self._conanfile = conanfile
        self.folder = folder
        self._excluded = excluded
        global_conf = conanfile._conan_helpers.global_conf  # noqa _conan_helpers
        conf_excluded = global_conf.get("core.scm:excluded", check_type=list)
        if conf_excluded:
            if excluded:
                c = ConfDefinition()
                c.loads(f"core.scm:excluded={excluded}")
                c.update_conf_definition(global_conf)
                self._excluded = c.get("core.scm:excluded", check_type=list)
            else:
                self._excluded = conf_excluded
        self._local_url = global_conf.get("core.scm:local_url", choices=["allow", "block"])

    def run(self, cmd, hidden_output=None):
        """
        Executes ``git <cmd>``

        :return: The console output of the command.
        """
        print_cmd = cmd if hidden_output is None else cmd.replace(hidden_output, "<hidden>")
        self._conanfile.output.info(f"RUN: git {print_cmd}", fg=Color.BRIGHT_BLUE)
        with chdir(self._conanfile, self.folder):
            # We tried to use self.conanfile.run(), but it didn't work:
            #  - when using win_bash, crashing because access to .settings (forbidden in source())
            #  - the ``conan source`` command, not passing profiles, buildenv not injected
            return check_output_runner("git {}".format(cmd)).strip()

    def get_commit(self, repository=False):
        """
        :param repository: By default gets the commit of the defined folder, use repo=True to get
                     the commit of the repository instead.
        :return: The current commit, with ``git rev-list HEAD -n 1 -- <folder>``.
            The latest commit is returned, irrespective of local not committed changes.
        """
        try:
            # commit = self.run("rev-parse HEAD") For the whole repo
            # This rev-list knows to capture the last commit for the folder
            # --full-history is needed to not avoid wrong commits:
            # https://github.com/conan-io/conan/issues/10971
            # https://git-scm.com/docs/git-rev-list#Documentation/git-rev-list.txt-Defaultmode
            path = '' if repository else '-- "."'
            commit = self.run(f'rev-list HEAD -n 1 --full-history {path}')
            return commit
        except Exception as e:
            raise ConanException("Unable to get git commit in '%s': %s" % (self.folder, str(e)))

    def get_remote_url(self, remote="origin"):
        """
        Obtains the URL of the remote git remote repository, with ``git remote -v``

        **Warning!**
        Be aware that This method will get the output from ``git remote -v``.
        If you added tokens or credentials to the remote in the URL, they will be exposed.
        Credentials shouldn’t be added to git remotes definitions, but using a credentials manager
        or similar mechanism. If you still want to use this approach, it is your responsibility
        to strip the credentials from the result.

        :param remote: Name of the remote git repository ('origin' by default).
        :return: URL of the remote git remote repository.
        """
        remotes = self.run("remote -v")
        for r in remotes.splitlines():
            name, url = r.split(maxsplit=1)
            if name == remote:
                url, _ = url.rsplit(None, 1)
                # if the url still has (fetch) or (push) at the end
                if url.endswith("(fetch)") or url.endswith("(push)"):
                    url, _ = url.rsplit(None, 1)
                if os.path.exists(url):  # Windows local directory
                    url = url.replace("\\", "/")
                return url

    def commit_in_remote(self, commit, remote="origin"):
        """
        Checks that the given commit exists in the remote, with ``branch -r --contains <commit>``
        and checking an occurrence of a branch in that remote exists.

        :param commit: Commit to check.
        :param remote: Name of the remote git repository ('origin' by default).
        :return: True if the given commit exists in the remote, False otherwise.
        """
        if not remote:
            return False
        # Potentially do two checks here.  If the clone is a shallow clone, then we won't be
        # able to find the commit.
        try:
            branches = self.run("branch -r --contains {}".format(commit))
            if "{}/".format(remote) in branches:
                return True
        except Exception as e:
            raise ConanException("Unable to check remote commit in '%s': %s" % (self.folder, str(e)))

        try:
            # This will raise if commit not present.
            self.run(f"fetch {remote} --refetch --dry-run {commit}")
            return True
        except (Exception,):
            # For example, if using an older git<2.36, see
            # https://github.com/conan-io/conan/issues/18470, then lets try the old approach
            try:
                # This will raise if commit not present.
                self.run("fetch {} --dry-run --depth=1 {}".format(remote, commit))
                return True
            except (Exception,):
                # Don't raise an error because it could fail for many more reasons.
                return False

    def is_dirty(self, repository=False):
        """
        Returns if the current folder is dirty, running ``git status -s``
        The ``Git(..., excluded=[])`` argument and the ``core.scm:excluded`` configuration will
        define file patterns to be skipped from this check.

        :param repository: By default checks if the current folder is dirty. If repository=True
                     it will check the root repository folder instead, not the current one.
        :return: True, if the current folder is dirty. Otherwise, False.
        """
        path = '' if repository else '.'
        status = self.run(f"status {path} --short --no-branch --untracked-files").strip()
        self._conanfile.output.debug(f"Git status:\n{status}")
        if not self._excluded:
            return bool(status)
        # Parse the status output, line by line, and match it with "_excluded"
        lines = [line.strip() for line in status.splitlines()]
        # line is of the form STATUS PATH, get the path by splitting
        # (Taking into account that STATUS is one word, PATH might be many)
        lines = [line.split(maxsplit=1)[1].strip('"') for line in lines if line]
        lines = [line for line in lines if not any(fnmatch.fnmatch(line, p) for p in self._excluded)]
        self._conanfile.output.debug(f"Filtered git status: {lines}")
        return bool(lines)

    def get_url_and_commit(self, remote="origin", repository=False):
        """
        This is an advanced method, that returns both the current commit, and the remote repository url.
        This method is intended to capture the current remote coordinates for a package creation,
        so that can be used later to build again from sources from the same commit. This is the behavior:

        * If the repository is dirty, it will raise an exception. Doesn’t make sense to capture coordinates
          of something dirty, as it will not be reproducible. If there are local changes, and the
          user wants to test a local conan create, should commit the changes first (locally, not push the changes).

        * If the repository is not dirty, but the commit doesn’t exist in the given remote, the method
          will return that commit and the URL of the local user checkout. This way, a package can be
          conan create created locally, testing everything works, before pushing some changes to the remote.

        * If the repository is not dirty, and the commit exists in the specified remote, it will
          return that commit and the url of the remote.

        **Warning!**
        Be aware that This method will get the output from ``git remote -v``.
        If you added tokens or credentials to the remote in the URL, they will be exposed.
        Credentials shouldn’t be added to git remotes definitions, but using a credentials manager
        or similar mechanism. If you still want to use this approach, it is your responsibility
        to strip the credentials from the result.

        :param remote: Name of the remote git repository ('origin' by default).
        :param repository: By default gets the commit of the defined folder, use repo=True to get
                     the commit of the repository instead.
        :return: (url, commit) tuple
        """
        dirty = self.is_dirty(repository=repository)
        if dirty:
            raise ConanException("Repo is dirty, cannot capture url and commit: "
                                 "{}".format(self.folder))
        commit = self.get_commit(repository=repository)
        url = self.get_remote_url(remote=remote)
        in_remote = self.commit_in_remote(commit, remote=remote)
        if in_remote:
            return url, commit
        if self._local_url == "block":
            raise ConanException(f"Current commit {commit} doesn't exist in remote {remote}\n"
                                 "Failing according to 'core.scm:local_url=block' conf")

        if self._local_url != "allow":
            self._conanfile.output.warning("Current commit {} doesn't exist in remote {}\n"
                                           "This revision will not be buildable in other "
                                           "computer".format(commit, remote))
        return self.get_repo_root(), commit

    def get_repo_root(self):
        """
        Get the current repository top folder with ``git rev-parse --show-toplevel``

        :return: Repository top folder.
        """
        folder = self.run("rev-parse --show-toplevel")
        return folder.replace("\\", "/")

    def clone(self, url, target="", args=None, hide_url=True):
        """
        Performs a ``git clone <url> <args> <target>`` operation, where target is the target directory.

        :param url: URL of remote repository.
        :param target: Target folder.
        :param args: Extra arguments to pass to the git clone as a list.
        :param hide_url: Hides the URL from the log output to prevent accidental
                     credential leaks. Can be disabled by passing ``False``.
        """
        args = args or []
        if os.path.exists(url):
            url = url.replace("\\", "/")  # Windows local directory
        mkdir(self.folder)
        self._conanfile.output.info("Cloning git repo")
        target_path = f'"{target}"' if target else ""  # quote in case there are spaces in path
        # Avoid printing the clone command, it can contain tokens
        self.run('clone "{}" {} {}'.format(url, " ".join(args), target_path),
                 hidden_output=url if hide_url else None)

    def fetch_commit(self, url, commit, hide_url=True):
        """
        Experimental: does a single commit fetch and checkout, instead of a full clone,
        should be faster.

        :param url: URL of remote repository.
        :param commit: The commit ref to checkout.
        :param hide_url: Hides the URL from the log output to prevent accidental
                     credential leaks. Can be disabled by passing ``False``.
        """
        if os.path.exists(url):
            url = url.replace("\\", "/")  # Windows local directory
        mkdir(self.folder)
        self._conanfile.output.info("Shallow fetch of git repo")
        self.run('init')
        self.run(f'remote add origin "{url}"', hidden_output=url if hide_url else None)
        self.run(f'fetch --depth 1 origin {commit}')
        self.run('checkout FETCH_HEAD')

    def checkout(self, commit):
        """
        Checkouts the given commit using ``git checkout <commit>``.

        :param commit: Commit to checkout.
        """
        self._conanfile.output.info("Checkout: {}".format(commit))
        self.run('checkout {}'.format(commit))

    def included_files(self):
        """
        Run ``git ls-files --full-name --others --cached --exclude-standard`` to the get the list
            of files not ignored by ``.gitignore``

        :return: List of files.
        """
        files = self.run("ls-files --full-name --others --cached --exclude-standard")
        files = files.splitlines()
        return files

    def coordinates_to_conandata(self, repository=False):
        """
        Capture the "url" and "commit" from the Git repo, calling ``get_url_and_commit()``, and then
        store those in the ``conandata.yml`` under the "scm" key. This information can be
        used later to clone and checkout the exact source point that was used to create this
        package, and can be useful even if the recipe uses ``exports_sources`` as mechanism to
        embed the sources.

        :param repository: By default gets the commit of the defined folder, use repository=True to get
                     the commit of the repository instead.
        """
        scm_url, scm_commit = self.get_url_and_commit(repository=repository)
        update_conandata(self._conanfile, {"scm": {"commit": scm_commit, "url": scm_url}})

    def checkout_from_conandata_coordinates(self):
        """
        Reads the "scm" field from the ``conandata.yml``, that must contain at least "url" and
        "commit" and then do a ``clone(url, target=".")``, ``fetch <commit>``, followed by a ``checkout(commit)``.
        """
        sources = self._conanfile.conan_data["scm"]
        self.clone(url=sources["url"], target=".", args=["--origin=origin"])
        self.run(f"fetch origin {sources['commit']}")
        self.checkout(commit=sources["commit"])
