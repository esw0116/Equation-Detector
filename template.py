def set_template(args):

    if args.template.find('SY') >= 0:
        args.work_type = 'Character'
        args.model = 'baseline'
        args.batch_size = 128
        args.num_epochs = 10
        args.decay_step = 1
        args.learning_rate = 2e-3
        args.gamma = 0.5

    elif args.template.find('DW') >= 0:
        args.work_type = 'Character'
        args.model = 'inception_model'
        args.batch_size = 64
        args.epochs = 5
        args.learning_rate = 2e-3
        args.decay_step = 1
        args.gamma = 0.5

    elif args.template.find('res') >= 0:
        args.work_type = 'Character'
        args.model = 'resnet'
        args.batch_size = 128
        args.num_epochs = 10
        args.learning_rate = 2e-3
        args.decay_step = 1
        args.gamma = 0.5

    elif args.template.find('RNN') >= 0:
        args.work_type = 'Expression'
        args.model = 'CRNN'
        args.batch_size = 32
        args.num_epochs = 5
        args.learning_rate = 4e-3
        args.decay_step = 1
        args.gamma = 0.5

    elif args.template.find('test') >= 0:
        args.work_type = 'Character'
        args.model = 'resnet'
        args.test_only = True
        args.load_path = '20180603_resnet_001'
        args.batch_size = 128
        args.num_epochs = 10
        args.decay_step = 1
        args.learning_rate = 2e-3
        args.gamma = 0.5