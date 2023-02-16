document.addEventListener('DOMContentLoaded', function () {
    adjustWidthforNavbarSelectors();
});

function adjustWidthforNavbarSelectors() {
    // adjust button size for navbar selectors
    var navbarSelectors = $('.sst-dropdown-navbar');
    navbarSelectors.each(function () {
        var dropdownButton = $(this).find('.btn');
        var maxWidth = dropdownButton.width();
        var dropdownItems = $(this).find('.dropdown-item');
        var itemWidth;
        dropdownItems.each(function () {
            itemWidth = $(this).text().length * 10;
            if (itemWidth > maxWidth) {
                maxWidth = itemWidth;
            }
        });
        dropdownButton.width(maxWidth);
    });
}
