const eloquaUrl = 'https://httpbingo.org/post'


$(document).ready(function () {
    // trigger without iframe
    // $('#newsletterTrigger').on('click', showForm);

    $('iframe').on('load', function() {
        $('iframe').contents().find('#newsletterTrigger').on('click', showForm);
    });

    function showForm() {
        fetch('_static/html/newsletter.html').then((response) => response.text()).then((text) => {
            const newsletter = $('<div>');
            newsletter.attr('id', 'newsletterModal');
            newsletter.addClass('newsletterContainer');

            const newsletterContent = $(text);
            newsletter.append(newsletterContent);
            $('body').prepend(newsletter);

            $('#newsletterEmail').focus();

            $('.modal-close').on('click', closeForm);
            $('#newsletterEmail').on('keyup', validate);

            $("#newsletterForm").submit(function(event) {
                event.preventDefault();
                const formHeight = $(this).outerHeight()
                $(this).removeClass('animated fade-up')
                $(this).animate({opacity: 0}, 200, 'linear', () => {
                    $.post(eloquaUrl, $(this).serialize())
                    .done(function(data) {
                        // ---------- debug request data
                        // console.log(data);
                        console.log('#############');
                        console.log('Origin: ' + data.headers['Origin'][0]);
                        console.log('Url: ' + data.url);
                        console.log('Form data:');
                        for (key in data.form) {
                            console.log(`-- ${key}: ${data.form[key]}`);
                        }
                        // ----------
                        displayMessage(formHeight, 'pass');
                    })
                    .fail(function(error) {
                        displayMessage(formHeight, 'error', error.status);
                    });
                });
            })
        })
    }

    function closeForm() {
        $('#newsletterModal').animate({opacity: 0}, 200, 'linear', function() {
            this.remove();
        });
    }

    function validate() {
        let value = $('#newsletterEmail').val();
        const emailPattern = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;
        if (emailPattern.test(value)) {
            $('#newsletterEmail').removeClass('failed');
            $('.newsletter-btn').prop('disabled', false);
        }
        else {
            $('#newsletterEmail').addClass('failed');
            $('.newsletter-btn').prop('disabled', true);
        }
    }

    function displayMessage(boxHeight, status, errorCode) {
        $('#newsletterForm').hide();
        let message = '';
        const messageBox = $('.message-box');
        const icon = $('<div class="fa-stack fa-2x">');
        const iconBackground = $('<i class="fas fa-square fa-stack-2x">');
        const iconMain = $('<i class="fas fa-stack-1x fa-inverse">');
        icon.append(iconBackground);
        icon.append(iconMain);
        messageBox.css({'height': boxHeight + 16, 'display': 'flex'});

        switch(status) {
            case 'pass':
                icon.css('color', '#708541');
                iconMain.addClass('fa-check-square');
                message = 'REGISTRATION SUCCESSFUL'
                break;
            case 'error':
                icon.css('color', '#C81326');
                iconMain.addClass('fa-window-close');
                switch(errorCode) {
                    case 400:
                        message = 'ALREADY REGISTERED';
                        break;
                    default:
                        message = 'REGISTRATION FAILED';
                        break;
                }
        }
        window.setTimeout(() => {
            messageBox.append(icon);
            messageBox.append(message);
        });
        window.setTimeout(closeForm, 1500);
    }
});
