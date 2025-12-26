
import fnmatch
import re
from functools import total_ordering

from conan.errors import ConanException
from conan.internal.model.version import Version
from conan.internal.util.dates import timestamp_to_str


@total_ordering
class RecipeReference:
    """ An exact (no version-range, no alias) reference of a recipe,
    it represents a reference of the form ``name/version[@user/channel][#revision][%timestamp]``.
    Should be enough to locate a recipe in the cache or in a server, and
    validation will be external to this class, at specific points (export, api, etc).
    """

    def __init__(self, name=None, version=None, user=None, channel=None, revision=None,
                 timestamp=None):
        """ The attributes should be regarded as immutable, and should not be modified by the user."""
        #: Name of the reference
        self.name: str = name
        if version is not None and not isinstance(version, Version):
            version = Version(version)
        #: Version of the reference
        self.version: Version = version  # This MUST be a version if we want to be able to order
        #: User of the reference, if any
        self.user = user
        #: Channel of the reference, if any
        self.channel = channel
        #: Revision of the reference, if any
        self.revision = revision
        #: Timestamp of the reference, if any
        self.timestamp = timestamp

    def copy(self):
        # Used for creating copy in lockfile-overrides mechanism
        return RecipeReference(self.name, self.version, self.user, self.channel, self.revision,
                               self.timestamp)

    def __repr__(self):
        """ long repr like pkg/0.1@user/channel#rrev%timestamp """
        result = self.repr_notime()
        if self.timestamp is not None:
            result += "%{}".format(self.timestamp)
        return result

    def repr_notime(self):
        result = self.__str__()
        if self.revision is not None:
            result += "#{}".format(self.revision)
        return result

    def repr_humantime(self):
        result = self.repr_notime()
        assert self.timestamp
        result += " ({})".format(timestamp_to_str(self.timestamp))
        return result

    def __str__(self):
        """ shorter representation, excluding the revision and timestamp """
        if self.name is None:
            return ""
        result = "/".join([self.name, str(self.version)])
        if self.user:
            result += "@{}".format(self.user)
        if self.channel:
            assert self.user
            result += "/{}".format(self.channel)
        return result

    def __lt__(self, ref):
        # The timestamp goes before the revision for ordering revisions chronologically
        # In theory this is enough for sorting
        # When no timestamp is given, it will always have lower priority, to avoid comparison
        # errors float <> None
        return (self.name, self.version, self.user or "", self.channel or "", self.timestamp or 0,
                self.revision or "") \
               < (ref.name, ref.version, ref.user or "", ref.channel or "", ref.timestamp or 0,
                  ref.revision or "")

    def __eq__(self, ref):
        # Timestamp doesn't affect equality.
        # This is necessary for building an ordered list of UNIQUE recipe_references for Lockfile
        if ref is None:
            return False
        # If one revision is not defined, they are equal
        if self.revision is not None and ref.revision is not None:
            return (self.name, self.version, self.user, self.channel, self.revision) == \
                   (ref.name, ref.version, ref.user, ref.channel, ref.revision)
        return (self.name, self.version, self.user, self.channel) == \
               (ref.name, ref.version, ref.user, ref.channel)

    def __hash__(self):
        # This is necessary for building an ordered list of UNIQUE recipe_references for Lockfile
        return hash((self.name, self.version, self.user, self.channel))

    @staticmethod
    def loads(rref):
        """ Instantiates an object from a string, in the form:
        ``name/version[@user/channel][#revision][%timestamp]``"""
        try:
            # timestamp
            tokens = rref.rsplit("%", 1)
            text = tokens[0]
            timestamp = float(tokens[1]) if len(tokens) == 2 else None

            # revision
            tokens = text.split("#", 1)
            ref = tokens[0]
            revision = tokens[1] if len(tokens) == 2 else None

            # name, version always here
            tokens = ref.split("@", 1)
            name, version = tokens[0].split("/", 1)
            assert name and version
            # user and channel
            if len(tokens) == 2 and tokens[1]:
                tokens = tokens[1].split("/", 1)
                user = tokens[0] if tokens[0] else None
                channel = tokens[1] if len(tokens) == 2 else None
            else:
                user = channel = None
            return RecipeReference(name, version, user, channel, revision, timestamp)
        except Exception:
            from conan.errors import ConanException
            raise ConanException(
                f"{rref} is not a valid recipe reference, provide a reference"
                f" in the form name/version[@user/channel]")

    def validate_ref(self, allow_uppercase=False):
        """ Check that the reference is valid, and raise a ``ConanException`` if not.
        """
        from conan.api.output import ConanOutput
        self_str = str(self)
        if self_str != self_str.lower():
            if not allow_uppercase:
                raise ConanException(f"Conan packages names '{self_str}' must be all lowercase")
            else:
                ConanOutput().warning(f"Package name '{self_str}' has uppercase, and has been "
                                      "allowed by temporary config. This will break in later 2.X")
        if len(self_str) > 200:
            raise ConanException(f"Package reference too long >200 {self_str}")
        if ":" in repr(self):
            raise ConanException(f"Invalid recipe reference '{repr(self)}' is a package reference")
        if not allow_uppercase:
            validation_pattern = re.compile(r"^[a-z0-9_][a-z0-9_+.-]{1,100}\Z")
        else:
            validation_pattern = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_+.-]{1,100}\Z")
        if validation_pattern.match(self.name) is None:
            raise ConanException(f"Invalid package name '{self.name}'")
        if validation_pattern.match(str(self.version)) is None:
            raise ConanException(f"Invalid package version '{self.version}'")
        if self.user and validation_pattern.match(self.user) is None:
            raise ConanException(f"Invalid package user '{self.user}'")
        if self.channel and validation_pattern.match(self.channel) is None:
            raise ConanException(f"Invalid package channel '{self.channel}'")

        # Warn if they use .+ in the name/user/channel, as it can be problematic for generators
        pattern = re.compile(r'[.+]')
        if pattern.search(self.name):
            ConanOutput().warning(f"Name containing special chars is discouraged '{self.name}'")
        if self.user and pattern.search(self.user):
            ConanOutput().warning(f"User containing special chars is discouraged '{self.user}'")
        if self.channel and pattern.search(self.channel):
            ConanOutput().warning(f"Channel containing special chars is discouraged "
                                  f"'{self.channel}'")

    def matches(self, pattern, is_consumer):
        """ fnmatches the reference against the provided pattern.

        :parameter str pattern: the pattern to match against, it can contain wildcards,
            and can start with ``!`` or ``~`` to negate the match.
            A special value of ``&`` will return a match only of ``is_consumer`` is ``True``
        :parameter bool is_consumer: if ``True``, the pattern ``&`` will match this reference.
        """
        negate = False
        if pattern.startswith("!") or pattern.startswith("~"):
            pattern = pattern[1:]
            negate = True

        no_user_channel = False
        if pattern.endswith("@"):  # it means we want to match only without user/channel
            pattern = pattern[:-1]
            no_user_channel = True
        elif "@#" in pattern:
            pattern = pattern.replace("@#", "#")
            no_user_channel = True

        condition = ((pattern == "&" and is_consumer) or
                     fnmatch.fnmatchcase(str(self), pattern) or
                     fnmatch.fnmatchcase(self.repr_notime(), pattern))
        if no_user_channel:
            condition = condition and not self.user and not self.channel
        if negate:
            return not condition
        return condition

    def partial_match(self, pattern):
        # Finds if pattern matches any of partial sums of tokens of conan reference
        tokens = [self.name, "/", str(self.version)]
        if self.user:
            tokens += ["@", self.user]
        if self.channel:
            tokens += ["/", self.channel]
        if self.revision:
            tokens += ["#", self.revision]
        partial = ""
        for token in tokens:
            partial += token
            if pattern.match(partial):
                return True


class PkgReference:

    def __init__(self, ref=None, package_id=None, revision=None, timestamp=None):
        self.ref = ref
        self.package_id = package_id
        self.revision = revision
        self.timestamp = timestamp  # float, Unix seconds UTC

    def __repr__(self):
        """ long repr like pkg/0.1@user/channel#rrev%timestamp """
        if self.ref is None:
            return ""
        result = repr(self.ref)
        if self.package_id:
            result += ":{}".format(self.package_id)
        if self.revision is not None:
            result += "#{}".format(self.revision)
        if self.timestamp is not None:
            result += "%{}".format(self.timestamp)
        return result

    def repr_notime(self):
        if self.ref is None:
            return ""
        result = self.ref.repr_notime()
        if self.package_id:
            result += ":{}".format(self.package_id)
        if self.revision is not None:
            result += "#{}".format(self.revision)
        return result

    def repr_humantime(self):
        result = self.repr_notime()
        assert self.timestamp
        result += " ({})".format(timestamp_to_str(self.timestamp))
        return result

    def __str__(self):
        """ shorter representation, excluding the revision and timestamp """
        if self.ref is None:
            return ""
        result = str(self.ref)
        if self.package_id:
            result += ":{}".format(self.package_id)
        return result

    def __lt__(self, ref):
        # The timestamp goes before the revision for ordering revisions chronologically
        raise Exception("WHO IS COMPARING PACKAGE REFERENCES?")
        # return (self.name, self.version, self.user, self.channel, self.timestamp, self.revision) \
        #       < (ref.name, ref.version, ref.user, ref.channel, ref._timestamp, ref.revision)

    def __eq__(self, other):
        # TODO: In case of equality, should it use the revision and timestamp?
        # Used:
        #    at "graph_binaries" to check: cache_latest_prev != pref
        #    at "installer" to check: if pkg_layout.reference != pref (probably just optimization?)
        #    at "revisions_test"
        return self.ref == other.ref and self.package_id == other.package_id and \
               self.revision == other.revision

    def __hash__(self):
        # Used in dicts of PkgReferences as keys like the cached nodes in the graph binaries
        return hash((self.ref, self.package_id, self.revision))

    @staticmethod
    def loads(pkg_ref):  # TODO: change this default to validate only on end points
        try:
            tokens = pkg_ref.split(":", 1)
            assert len(tokens) == 2
            ref, pkg_id = tokens

            ref = RecipeReference.loads(ref)

            # timestamp
            tokens = pkg_id.rsplit("%", 1)
            text = tokens[0]
            timestamp = float(tokens[1]) if len(tokens) == 2 else None

            # revision
            tokens = text.split("#", 1)
            package_id = tokens[0]
            revision = tokens[1] if len(tokens) == 2 else None

            return PkgReference(ref, package_id, revision, timestamp)
        except Exception:
            raise ConanException(
                f"{pkg_ref} is not a valid package reference, provide a reference"
                f" in the form name/version[@user/channel:package_id]")
