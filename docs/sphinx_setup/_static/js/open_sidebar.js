$(document).ready(function() {
    const labels = $( "ul.bd-sidenav > li > label" );
    for(let i = 0; i < labels.length; i++){
        labels[i].classList.remove("rotate");
    }
})