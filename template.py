def set_template(args):

    if args.template.find('SY') >= 0:
        args.work_type = 'Expression'
        args.model = 'CRNNv3'
        args.cnn_model = 'resnet'
        args.batch_size = 64
        args.num_epochs = 20
        args.learning_rate = 2e-3
        args.decay_step = 2
        args.gamma = 0.5
        args.embed_size = 512
        args.hidden_size = 512
        args.num_layers = 1

    elif args.template.find('DW') >= 0:
        args.work_type = 'Character'
        args.model = 'baseline'
        args.batch_size = 64
        args.epochs = 20
        args.learning_rate = 2e-3
        args.decay_step = 1
        args.gamma = 0.5

    elif args.template.find('res') >= 0:
        args.work_type = 'Character'
        args.model = 'resnet'
        args.load_path = '20180606_resnet_001'
        args.batch_size = 128
        args.num_epochs = 10
        args.learning_rate = 2e-3
        args.decay_step = 2
        args.gamma = 0.5

    elif args.template.find('RNN') >= 0:
        args.work_type = 'Expression'
        args.model = 'CRNN'
        args.cnn_model = 'resnet'
        args.batch_size = 64
        args.num_epochs = 20
        args.learning_rate = 2e-3
        args.decay_step = 2
        args.gamma = 0.5
        args.embed_size = 1024
        args.hidden_size = 1024
        args.num_layers = 1


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

    elif args.template.find('parameter_tuning_rnn') >= 0:
        args.work_type = 'Expression'
        args.model = 'CRNN'
        args.cnn_model = 'resnet'
        args.batch_size = 64
        args.num_epochs = 30
        args.learning_rate = 2e-3
        args.decay_step = 3
        args.gamma = 0.5
        args.embed_size = 1024
        args.hidden_size = 1024
        args.num_layers = 1
        