// submits data to the server and MTurk's server if the experiment runs on MTurk
// takes three arguments:
// 1) isMTurk - boolean; true if the experiment runs on MTurk
// 2) contactEmail - string
// 3) trials
var submitResults = function(contactEmail, submissionURL, data) {
    // set a default contact email
    contactEmail =
        typeof contactEmail !== "undefined" ? contactEmail : "exprag@gmail.com";

    $.ajax({
        type: "POST",
        url: submissionURL,
        crossDomain: true,
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function(responseData, textStatus, jqXHR) {
            console.log(textStatus);

            $(".warning-message").addClass("nodisplay");
            $(".thanks-message").removeClass("nodisplay");
            $(".extra-message").removeClass("nodisplay");

            if (config_deploy.is_MTurk) {
                // submits to MTurk's server if isMTurk = true
                setTimeout(function() {
                    submitToMTurk(data), 500;
                });
            }
        },
        error: function(responseData, textStatus, errorThrown) {
            // There is this consideration about whether we should still allow such a submission that failed on our side to proceed on submitting to MTurk. Maybe we should after all.
            if (config_deploy.is_MTurk) {
                // For now we still use the original turk.submit to inform MTurk that the experiment has finished.
                // Stela might have another implementation which submits only the participant id.
                // Not notifying the user yet since it might cause confusion. The webapp should report errors.

                // submits to MTurk's server if isMTurk = true
                submitToMTurk(data);

                // shows a thanks message after the submission
                $(".thanks-message").removeClass("nodisplay");
            } else {
                // It seems that this timeout (waiting for the server) is implemented as a default value in many browsers, e.g. Chrome. However it is really long (1 min) so timing out shouldn't be such a concern.
                if (textStatus == "timeout") {
                    alert(
                        "Oops, the submission timed out. Please try again. If the problem persists, please contact " +
                        contactEmail +
                        ", including your ID"
                    );
                } else {
                    alert(
                        "Oops, the submission failed. The server says: " +
                        responseData.responseText +
                        "\nPlease try again. If the problem persists, please contact " +
                        contactEmail +
                        "with this error message, including your ID"
                    );
                }
            }
        }
    });
};

// submits to MTurk's servers
// and the correct url is given in config.MTurk_server
var submitToMTurk = function() {
    var form = $("#mturk-submission-form");
    form.submit();
};