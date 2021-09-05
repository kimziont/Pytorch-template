def get_scheduler_args(config, lr_scheduler):
    _backbone_scheduler = {
    'StepLR': {
            "step_size": 50,
            "gamma": 0.1
            },
    'LambdaLR': {
            "lr_lambda": 0.95,
            },
    'MultiStepLR': {
            "milestones": [5, 10, 15, 20, 22, 24, 26, 28, 30],
            "gamma": 0.7
            },
    'ExponentialLR': {
            "gamma": 0.5
            },

    }, 
    
    if lr_scheduler in _backbone_scheduler[0]:
        return _backbone_scheduler[0][lr_scheduler]
    # optimizer in _custom_optimizer:
    else:
        pass