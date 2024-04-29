# Copyright (c) OpenMMLab. All rights reserved.


def longalpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'input': f"{example['instruction']}",
                'output': example['output']
            }]
        }
