$(document).ready(function() {
    var collapsible_sections = $('div.collapsible-section');
    $(collapsible_sections).wrap('<details class="col-sect-details sphinx-bs dropdown card mb-3"></details>');
    $('.col-sect-details').prepend('<summary class="summary-title card-header">Click to expand</summary>');
});