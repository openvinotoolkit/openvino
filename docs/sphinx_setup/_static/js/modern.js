$(document).ready(function() {
    const elems = $( 'details.sd-dropdown' );
    for(let i = 0; i < elems.length; i++){
        elems[i].style.cssText = 'box-shadow: none !important; border: 1px !important;'
    }


    const admonitions = $( '.admonition' );
    for(let i = 0; i < admonitions.length; i++){
        admonitions[i].style.cssText = 'box-shadow: none !important; border-radius:0px !important; '
    }
})