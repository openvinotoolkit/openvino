class RestRoutes(object):
    ping = "ping"
    common_search = "conans/search"
    common_authenticate = "users/authenticate"
    common_check_credentials = "users/check_credentials"

    def __init__(self):
        self.base = 'conans'

    @property
    def recipe(self):
        return self.base + '/{name}/{version}/{username}/{channel}'

    @property
    def recipe_latest(self):
        return '%s/latest' % self.recipe

    @property
    def recipe_revision(self):
        return '%s/revisions/{revision}' % self.recipe

    @property
    def recipe_revision_files(self):
        return '%s/files' % self.recipe_revision

    @property
    def recipe_revisions(self):
        return '%s/revisions' % self.recipe

    @property
    def recipe_revision_file(self):
        return '%s/files/{path}' % self.recipe_revision

    @property
    def packages_revision(self):
        return '%s/packages' % self.recipe_revision

    @property
    def package_recipe_revision(self):
        """Route for a package specifying the recipe revision but not the package revision"""
        return '%s/{package_id}' % self.packages_revision

    @property
    def package_revisions(self):
        return '%s/revisions' % self.package_recipe_revision

    @property
    def package_revision(self):
        return '%s/{p_revision}' % self.package_revisions

    @property
    def package_revision_files(self):
        return '%s/files' % self.package_revision

    @property
    def package_revision_latest(self):
        return '%s/latest' % self.package_recipe_revision

    @property
    def package_revision_file(self):
        return '%s/files/{path}' % self.package_revision

    @property
    def common_search_packages(self):
        return "%s/search" % self.recipe

    @property
    def common_search_packages_revision(self):
        return "%s/search" % self.recipe_revision
