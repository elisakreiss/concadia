// when the DOM is created and JavaScript code can run safely,
// the experiment initialisation is called
$("document").ready(function() {
    exp.init();
    // prevent scrolling when space is pressed (firefox does it)
    window.onkeydown = function(e) {
        if (e.keyCode == 32 && e.target == document.body) {
            e.preventDefault();
        }
    };
});

// empty shell for 'exp' object; to be filled with life by the init() function
var exp = {};

exp.init = function() {
    // allocate storage room for global data, trial data, and trial info
    this.global_data = {};
    this.trial_data = [];
    this.trial_info = {};

    // record current date and time
    this.global_data.startDate = Date();
    this.global_data.startTime = Date.now();

    // call user-defined costumization function
    this.customize();

    // flatten views_seq after possible 'loop' insertions
    this.views_seq = _.flatten(this.views_seq);
    // create Progress Bar/s
    this.progress = this.initProgressBar();
    this.progress.add();

    // insert a Current Trial counter for each view
    _.map(this.views_seq, function(i) {
        i.CT = 0;
    });

    // initialize procedure
    this.currentViewCounter = 0;
    this.currentTrialCounter = 0;
    this.currentTrialInViewCounter = 0;
    this.currentView = this.findNextView();

    // user does not (should not) change the following information
    // checks the config _deploy.deployMethod is MTurk or MTurkSandbox,
    // sets the submission url to MTukr's servers
    config_deploy.MTurk_server =
        config_deploy.deployMethod == "MTurkSandbox"
            ? "https://workersandbox.mturk.com/mturk/externalSubmit" // URL for MTurk sandbox
            : config_deploy.deployMethod == "MTurk"
                ? "https://www.mturk.com/mturk/externalSubmit" // URL for live HITs on MTurk
                : ""; // blank if deployment is not via MTurk
    // if the config_deploy.deployMethod is not debug, then liveExperiment is true
    config_deploy.liveExperiment = config_deploy.deployMethod !== "debug";
    config_deploy.is_MTurk = config_deploy.MTurk_server !== "";
    config_deploy.submissionURL =
        config_deploy.deployMethod == "localServer"
            ? "http://localhost:4000/api/submit_experiment/" +
              config_deploy.experimentID
            : config_deploy.serverAppURL + config_deploy.experimentID;
    console.log("deployMethod: " + config_deploy.deployMethod);
    console.log("live experiment: " + config_deploy.liveExperiment);
    console.log("runs on MTurk: " + config_deploy.is_MTurk);
    console.log("MTurk server: " + config_deploy.MTurk_server);
};

// navigation through the views and steps in each view;
// shows each view (in the order defined in 'config_general') for
// the given number of steps (as defined in the view's 'trial' property)
exp.findNextView = function() {
    var currentView = this.views_seq[this.currentViewCounter];
    if (this.currentTrialInViewCounter < currentView.trials) {
        currentView.render(currentView.CT, this.currentTrialInViewCounter);
    } else {
        this.currentViewCounter++;
        currentView = this.views_seq[this.currentViewCounter];
        this.currentTrialInViewCounter = 0;
        currentView.render(currentView.CT);
    }
    // increment counter for how many trials we have seen of THIS view during THIS occurrence of it
    this.currentTrialInViewCounter++;
    // increment counter for how many trials we have seen in the whole experiment
    this.currentTrialCounter++;
    // increment counter for how many trials we have seen of THIS view during the whole experiment
    currentView.CT++;
    if (currentView.hasProgressBar) {
        this.progress.update();
    }

    return currentView;
};

// submits the data
exp.submit = function() {
    // adds columns with NA values
    var addEmptyColumns = function(trialData) {
        var columns = [];

        for (var i = 0; i < trialData.length; i++) {
            for (var prop in trialData[i]) {
                if (
                    trialData[i].hasOwnProperty(prop) &&
                    columns.indexOf(prop) === -1
                ) {
                    columns.push(prop);
                }
            }
        }

        for (var j = 0; j < trialData.length; j++) {
            for (var k = 0; k < columns.length; k++) {
                if (!trialData[j].hasOwnProperty(columns[k])) {
                    trialData[j][columns[k]] = "NA";
                }
            }
        }

        return trialData;
    };

    var formatDebugData = function(flattenedData) {
        var output = "<table id = 'debugresults'>";

        var t = flattenedData[0];

        output += "<thead><tr>";

        for (var key in t) {
            if (t.hasOwnProperty(key)) {
                output += "<th>" + key + "</th>";
            }
        }

        output += "</tr></thead>";

        output += "<tbody><tr>";

        var entry = "";

        for (var i = 0; i < flattenedData.length; i++) {
            var currentTrial = flattenedData[i];
            for (var k in t) {
                if (currentTrial.hasOwnProperty(k)) {
                    entry = String(currentTrial[k]);
                    output += "<td>" + entry.replace(/ /g, "&nbsp;") + "</td>";
                }
            }

            output += "</tr>";
        }

        output += "</tbody></table>";

        return output;
    };

    var flattenData = function(data) {
        var trials = data.trials;
        delete data.trials;

        // The easiest way to avoid name clash is just to check the keys one by one and rename them if necessary.
        // Though I think it's also the user's responsibility to avoid such scenarios...
        var sample_trial = trials[0];
        for (var trial_key in sample_trial) {
            if (sample_trial.hasOwnProperty(trial_key)) {
                if (data.hasOwnProperty(trial_key)) {
                    // Much easier to just operate it once on the data, since if we also want to operate on the trials we'd need to loop through each one of them.
                    var new_data_key = "glb_" + trial_key;
                    data[new_data_key] = data[trial_key];
                    delete data[trial_key];
                }
            }
        }

        var out = _.map(trials, function(t) {
            // Here the data is the general informatoin besides the trials.
            return _.merge(t, data);
        });
        return out;
    };

    // construct data object for output
    var data = {
        experiment_id: config_deploy.experimentID,
        trials: addEmptyColumns(exp.trial_data)
    };

    // parses the url to get the assignmentId and workerId
    var getHITData = function() {
        var url = window.location.href;
        var qArray = url.split("?");
        qArray = qArray[1].split("&");
        var HITData = {};

        for (var i = 0; i < qArray.length; i++) {
            HITData[qArray[i].split("=")[0]] = qArray[i].split("=")[1];
        }

        console.log(HITData);
        return HITData;
    };

    // add more fields depending on the deploy method
    if (config_deploy.is_MTurk) {
        var HITData = getHITData();
        data["assignment_id"] = HITData["assignmentId"];
        data["worker_id"] = HITData["workerId"];
        data["hit_id"] = HITData["hitId"];

        // creates a form with assignmentId input for the submission ot MTurk
        var form = jQuery("<form/>", {
            id: "mturk-submission-form",
            action: config_deploy.MTurk_server
        }).appendTo(".thanks-templ");
        var dataForMTurk = jQuery("<input/>", {
            type: "hidden",
            name: "data",
            value: "done"
        }).appendTo(form);
        // MTurk expects a key 'assignmentId' for the submission to work,
        // that is why is it not consistent with the snake case that the other keys have
        var assignmentId = jQuery("<input/>", {
            type: "hidden",
            name: "assignmentId",
            value: HITData["assignmentId"]
        }).appendTo(form);
    } else if (config_deploy.deployMethod === "Prolific") {
        console.log();
    } else if (config_deploy.deployMethod === "directLink") {
        console.log();
    } else if (config_deploy.deployMethod === "debug") {
        console.log();
    } else {
        console.log("no such config_deploy.deployMethod");
    }

    // merge in global data accummulated so far
    // this could be unsafe if 'exp.global_data' contains keys used in 'data'!!
    data = _.merge(exp.global_data, data);

    // if the experiment is set to live (see config.js liveExperiment)
    // the results are sent to the server
    // if it is set to false
    // the results are displayed on the thanks slide
    if (config_deploy.liveExperiment) {
        console.log("submits");
        //submitResults(config_deploy.contact_email, config_deploy.submissionURL, data);
        submitResults(
            config_deploy.contact_email,
            config_deploy.submissionURL,
            flattenData(data)
        );
    } else {
        // hides the 'Please do not close the tab.. ' message in debug mode
        console.log(data);
        var flattenedData = flattenData(data);
        $(".warning-message").addClass("nodisplay");
        jQuery("<h3/>", {
            text: "Debug Mode"
        }).appendTo($(".view"));
        jQuery("<div/>", {
            class: "debug-results",
            html: formatDebugData(flattenedData)
        }).appendTo($(".view"));
        createCSVForDownload(flattenedData);
    }
};

var createCSVForDownload = function(flattenedData) {
    var csvOutput = "";

    var t = flattenedData[0];

    for (var key in t) {
        if (t.hasOwnProperty(key)) {
            csvOutput += '"' + String(key) + '",';
        }
    }
    csvOutput += "\n";
    for (var i = 0; i < flattenedData.length; i++) {
        var currentTrial = flattenedData[i];
        for (var k in t) {
            if (currentTrial.hasOwnProperty(k)) {
                csvOutput += '"' + String(currentTrial[k]) + '",';
            }
        }
        csvOutput += "\n";
    }

    var blob = new Blob([csvOutput], {
        type: "text/csv"
    });
    if (window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveBlob(blob, "results.csv");
    } else {
        jQuery("<a/>", {
            class: "button download-btn",
            html: "Download the results as CSV",
            href: window.URL.createObjectURL(blob),
            download: "results.csv"
        }).appendTo($(".view"));
    }
};

var processTrialsData = function(rows) {
    var toReturn = [];
    var headers = rows[0];
    for (var indexTrial = 1; indexTrial < rows.length; indexTrial++) {
        var thisTrial = {};
        for (var indexKey = 0; indexKey < headers.length; indexKey++) {
            thisTrial[headers[indexKey]] = rows[indexTrial][indexKey];
        }
        toReturn.push(thisTrial);
    }
    return toReturn;
};

var prepareDataFromCSV = function(practiceTrialsFile, trialsFile) {
    var data = {
        out: [] // mandatory field to store results in during experimental trials
    };

    // Need to use such a callback since AJAX is async.
    var addToContainer = function(container, name, results) {
        container[name] = results;
    };

    $.ajax({
        url: practiceTrialsFile,
        dataType: "text",
        crossDomain: true,
        success: function(file, _, jqXHR) {
            addToContainer(
                data,
                "practice_trials",
                processTrialsData(CSV.parse(file))
            );
        }
    });

    $.ajax({
        url: trialsFile,
        dataType: "text",
        crossDomain: true,
        success: function(file, textStatus, jqXHR) {
            addToContainer(
                data,
                "trials",
                _.shuffle(processTrialsData(CSV.parse(file)))
            );
        }
    });

    return data;
};

var loop = function(arr, count, shuffleFlag) {
    return _.flatMapDeep(_.range(count), function(i) {
        return arr;
    });
};

var loopShuffled = function(arr, count) {
    return _.flatMapDeep(_.range(count), function(i) {
        return _.shuffle(arr);
    });
};
