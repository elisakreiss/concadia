// customize the experiment by specifying a view order and a trial structure
exp.customize = function() {
    // record current date and time in global_data
    this.global_data.startDate = Date();
    this.global_data.startTime = Date.now();
    // specify view order
    this.views_seq = [
        botcaptcha,
        intro,
        main,
        postTest,
        thanks
    ];

    main_trials = _.shuffle(main_trials)[0]
    main_trials = main_trials.slice(270)
    console.log(main_trials)

    // prepare information about trials (procedure)
    cond = _.shuffle([
        'caption', 'caption', 'caption', 'caption', 'caption', 
        'caption', 'caption', 'caption', // 'caption', 'caption', 
        'description', 'description', 'description', 'description', 'description', 
        'description', 'description', 'description', // 'description', 'description',
        'generated_descr', 'generated_descr', 'generated_descr', 'generated_descr', 'generated_descr', 
        'generated_descr', 'generated_descr', 'generated_descr', // 'generated_descr', 'generated_descr', 
        'generated_capt', 'generated_capt', 'generated_capt', 'generated_capt', 'generated_capt', 
        'generated_capt', 'generated_capt', 'generated_capt' // , 'generated_capt', 'generated_capt'
    ])
    console.log(cond)
    for (i = 0; i < main_trials.length; i++) {
        main_trials[i]['condition'] = cond.pop()
    }
    console.log(main_trials)
    console.log("attention_checks")
    console.log(attention_checks)

    // add attention checks
    main_trials.push(...attention_checks);

    // randomize main trial order, but keep practice trial order fixed
    this.trial_info.main_trials = _.shuffle(main_trials);
    console.log("Number of stimuli");
    console.log(main_trials.length);
    console.log(this.trial_info.main_trials);

    // sample question order
    questions = _.shuffle(["replacement", "learn"])
    this.trial_info.q1 = questions.pop()
    this.trial_info.q2 = questions.pop()

    // adds progress bars to the views listed
    // view's name is the same as object's name
    this.progress_bar_in = ["main"];
    // this.progress_bar_in = ['practice', 'main'];
    // styles: chunks, separate or default
    this.progress_bar_style = "default";
    // the width of the progress bar or a single chunk
    this.progress_bar_width = 100;
};
