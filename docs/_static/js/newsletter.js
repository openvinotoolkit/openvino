const eloquaUrl = 'https://httpbin.org/status/200'


$(document).ready(function () {
    $('#newsletterTrigger').on('click', showForm);

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
                        console.log(data);
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
        messageBox.css({'height': boxHeight + 16, 'display': 'flex'});
        
        switch(status) {
            case 'pass':
                messageBox.css('color', 'rgba(3, 163, 0, 1)')
                message = 'REGISTRATION SUCCESSFUL'
                break;
            case 'error':
                messageBox.css('color', 'rgba(201, 0, 0, 1)')
                switch(errorCode) {
                    case 400:
                        message = 'ALREADY REGISTERED';
                        break;
                    default:
                        message = 'REGISTRATION FAILED';
                        break;
                }
        }
        messageBox.append(message);
        window.setTimeout(closeForm, 1500);
    }
});
