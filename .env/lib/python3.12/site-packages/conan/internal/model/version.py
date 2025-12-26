from functools import total_ordering
from typing import Optional

from conan.errors import ConanException


@total_ordering
class _VersionItem:
    """ a single "digit" in a version, like X.Y.Z all X and Y and Z are VersionItems
    They can be int or strings
    """
    def __init__(self, item):
        try:
            self._v = int(item)
        except ValueError:
            self._v = item

    @property
    def value(self):
        return self._v

    def __str__(self):
        return str(self._v)

    def __add__(self, other):
        # necessary for the "bump()" functionality. Other aritmetic operations are missing
        return self._v + other

    def __eq__(self, other):
        if not isinstance(other, _VersionItem):
            other = _VersionItem(other)
        return self._v == other._v

    def __hash__(self):
        return hash(self._v)

    def __lt__(self, other):
        """
        @type other: _VersionItem
        """
        if not isinstance(other, _VersionItem):
            other = _VersionItem(other)
        try:
            return self._v < other._v
        except TypeError:
            return str(self._v) < str(other._v)


@total_ordering
class Version:
    """
    This is NOT an implementation of semver, as users may use any pattern in their versions.
    It is just a helper to parse "." or "-" and compare taking into account integers when possible
    """
    def __init__(self, value, qualifier=False):
        value = str(value)
        self._value = value
        self._build = None
        self._pre = None
        self._qualifier = qualifier  # it is a prerelease or build qualifier, not a main version

        if not qualifier:
            items = value.rsplit("+", 1)  # split for build
            if len(items) == 2:
                value, build = items
                self._build = Version(build, qualifier=True)  # This is a nested version by itself

            # split for pre-release, from the left, semver allows hyphens in identifiers :(
            items = value.split("-", 1)
            if len(items) == 2:
                value, pre = items
                self._pre = Version(pre, qualifier=True)  # This is a nested version by itself

        items = value.split(".")
        items = [_VersionItem(item) for item in items]
        self._items = tuple(items)
        while items and items[-1].value == 0:
            del items[-1]
        self._nonzero_items = tuple(items)

    def bump(self, index):
        """
        :meta private:
            Bump the version
            Increments by 1 the version field at the specified index, setting to 0 the fields
            on the right.
            2.5 => bump(1) => 2.6
            1.5.7 => bump(0) => 2.0.0

        :param index:
        """
        # this method is used to compute version ranges from tilde ~1.2 and caret ^1.2.1 ranges
        # TODO: at this moment it only works for digits, cannot increment pre-release or builds
        # better not make it public yet, keep it internal
        items = list(self._items[:index])
        try:
            items.append(self._items[index]+1)
        except TypeError:
            raise ConanException(f"Cannot bump '{self._value} version index {index}, not an int")
        items.extend([0] * (len(items) - index - 1))
        v = ".".join(str(i) for i in items)
        # prerelease and build are dropped while bumping digits
        return Version(v)

    def upper_bound(self, index):
        items = list(self._items[:index])
        try:
            items.append(self._items[index] + 1)
        except TypeError:
            raise ConanException(f"Cannot bump '{self._value} version index {index}, not an int")
        items.extend([0] * (len(items) - index - 1))
        v = ".".join(str(i) for i in items)
        v += "-"  # Exclude prereleases
        return Version(v)

    @property
    def pre(self):
        return self._pre

    @property
    def build(self):
        return self._build

    @property
    def main(self):
        return self._items

    @property
    def major(self):
        try:
            return self.main[0]
        except IndexError:
            return None

    @property
    def minor(self):
        try:
            return self.main[1]
        except IndexError:
            return None

    @property
    def patch(self):
        try:
            return self.main[2]
        except IndexError:
            return None

    @property
    def micro(self):
        try:
            return self.main[3]
        except IndexError:
            return None

    def __str__(self):
        return self._value

    def __repr__(self):
        return self._value

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, Version):
            other = Version(other, self._qualifier)

        return (self._nonzero_items, self._pre, self._build) ==\
               (other._nonzero_items, other._pre, other._build)

    def __hash__(self):
        return hash((self._nonzero_items, self._pre, self._build))

    def __lt__(self, other):
        if other is None:
            return False
        if not isinstance(other, Version):
            other = Version(other)

        if self._pre:
            if other._pre:  # both are pre-releases
                return (self._nonzero_items, self._pre, self._build) < \
                       (other._nonzero_items, other._pre, other._build)
            else:  # Left hand is pre-release, right side is regular
                if self._nonzero_items == other._nonzero_items:  # Problem only happens if both equal
                    return True
                else:
                    return self._nonzero_items < other._nonzero_items
        else:
            if other._pre:  # Left hand is regular, right side is pre-release
                if self._nonzero_items == other._nonzero_items:  # Problem only happens if both equal
                    return False
                else:
                    return self._nonzero_items < other._nonzero_items
            else:  # None of them is pre-release
                return (self._nonzero_items, self._build) < (other._nonzero_items, other._build)

    def in_range(self, version_range: str, resolve_prerelease: Optional[bool] = None):
        """ Check if the version is in the specified range """
        from conan.internal.model.version_range import VersionRange
        return VersionRange(version_range).contains(self, resolve_prerelease=resolve_prerelease)
