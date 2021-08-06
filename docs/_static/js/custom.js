$(document).ready(function() {
    var collapsible_sections = $('div.collapsible-section');
    $(collapsible_sections).wrap('<details class="col-sect-details"></details>');
    $('.col-sect-details').prepend('<summary>Click to expand</summary>');
});