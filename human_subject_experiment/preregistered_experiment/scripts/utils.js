exp.initProgressBar = function() {
    var progressBar = {};
    var totalProgressParts = 0;
    var progressTrials = 0;
    if (exp.progress_bar_style === "chunks" || "separate") {
        var totalProgressChunks = 0;
        var filledChunks = 0;
        var fillChunk = false;
    }

    // creates progress bar element(s) and adds it(them) to the view
    var addToDOM = function() {
        var bar;
        var i;
        var view = $(".view");
        var barWidth = exp.progress_bar_width;
        var clearfix = jQuery("<div/>", {
            class: "clearfix"
        });
        var container = jQuery("<div/>", {
            class: "progress-bar-container"
        });
        view.css("padding-top", 30);
        view.prepend(clearfix);
        view.prepend(container);

        if (exp.progress_bar_style === "chunks") {
            for (i = 0; i < totalProgressChunks; i++) {
                bar = jQuery("<div/>", {
                    class: "progress-bar"
                });
                bar.css("width", barWidth);
                container.append(bar);
            }
        } else if (exp.progress_bar_style === "separate") {
            bar = jQuery("<div/>", {
                class: "progress-bar"
            });
            bar.css("width", barWidth);
            container.append(bar);
        } else if (exp.progress_bar_style === "default") {
            bar = jQuery("<div/>", {
                class: "progress-bar"
            });
            bar.css("width", barWidth);
            container.append(bar);
        } else {
            console.debug(
                'exp.progress_bar_style can be set to "default", "separate" or "chunks"'
            );
        }
    };

    // adds progress bar(s) to the specified experiment.js, this.progress_bar_in
    progressBar.add = function() {
        for (var i = 0; i < exp.views_seq.length; i++) {
            for (var j = 0; j < exp.progress_bar_in.length; j++) {
                if (exp.views_seq[i].name === exp.progress_bar_in[j]) {
                    totalProgressChunks++;
                    totalProgressParts += exp.views_seq[i].trials;
                    exp.views_seq[i].hasProgressBar = true;
                }
            }
        }
    };

    // updates the progress of the progress bar
    // creates a new progress bar(s) for each view that has it and updates it
    progressBar.update = function() {
        addToDOM();

        var progressBars = $(".progress-bar");
        var div, filledElem, filledPart;

        if (exp.progress_bar_style === "default") {
            div = $(".progress-bar").width() / totalProgressParts;
            filledPart = progressTrials * div;
        } else {
            div =
                $(".progress-bar").width() /
                exp.views_seq[exp.currentViewCounter].trials;
            filledPart = (exp.currentTrialInViewCounter - 1) * div;
        }

        filledElem = jQuery("<span/>", {
            id: "filled"
        }).appendTo(progressBars[filledChunks]);

        $("#filled").css("width", filledPart);
        progressTrials++;

        if (exp.progress_bar_style === "chunks") {
            if (fillChunk === true) {
                filledChunks++;
                fillChunk = false;
            }

            if (filledElem.width() === $(".progress-bar").width() - div) {
                fillChunk = true;
            }

            for (var i = 0; i < filledChunks; i++) {
                progressBars[i].style.backgroundColor = "#5187BA";
            }
        }
    };

    return progressBar;
};
