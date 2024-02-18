$(document).ready(function() {
    const elems = $( "ul.bd-sidenav > li > input" );
    for(let i = 0; i < elems.length; i++){
        elems[i].setAttribute("checked", "checked");
    }
})