def set_template(args):

    if args.template.find('SY') >= 0:
        args.work_type = 'Character'
        args.model = 'baseline'
        args.batch_size = 32
        args.num_batches = 16000
        args.decay_step = 5000
        args.learning_rate = 2e-3
        args.gamma = 0.5

    elif args.template.find('DW') >= 0:
        args.work_type = 'Character'
        # Add your own argparse
