$(document).ready(function() {
    const labels = $( "ul.bd-sidenav > li > label" );
    for(let i = 0; i < labels.length; i++){
        labels[i].classList.remove("rotate");
    }

    const menus = $( "ul.bd-sidenav > li > a" );
    for(let i = 0; i < menus.length; i++){
        menus[i].classList.add("bold");
    }
})