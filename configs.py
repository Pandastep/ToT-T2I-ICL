#configs.py
task_types = [
    'color_object',  
    'style_object', 
    'action_animal', 
    'texture_object',
    'background_animal',
]

item2word = {
    '3d': 'wireframe',
    'park': 'amusement park',
}

# reverse item2word
word2item = {v: k for k, v in item2word.items()}

item_dict = {
    'color': ['yellow', 'white', 'red', 'purple', 'pink', 'orange', 'green', 'brown', 'blue', 'black'],
    'object': ['leaf', 'hat', 'cup', 'chair', 'car', 'box', 'book', 'ball', 'bag', 'apple'],
    'weather': ['tornado', 'thunder', 'sunny', 'snowy', 'sandstorm', 'rainy', 'rainbow', 'hailstorm', 'foggy', 'aurora'],
    'animal': ['zebra', 'tiger', 'sheep', 'pig', 'monkey', 'lion', 'dog', 'cow', 'cat', 'bird'],
    'style': ['watercolor', 'sketch', 'pixel', 'origami', 'lego', 'icon', 'graffiti', 'futuristic', '3d', 'old'],
    'action': ['swim', 'sleep', 'sing', 'run', 'read', 'fly', 'eat', 'drink', 'cry', 'angry'],
    'background': ['beach', 'desert', 'glacier', 'volcano', 'park', 'gym', 'waterfall', 'space', 'cave', 'seafloor'],
    'texture': ['wood', 'wicker', 'sequined', 'plastic', 'paper', 'metal', 'leather', 'lace', 'denim', 'ceramic'],
}

task_dataframe = {
    1: {
        'task_name': 'Color-I',
        'task_type': 'color_object',
        'x_space': 'color',
        'theta_space': 'object',
        'x_list': item_dict['color'],
        'theta_list': item_dict['object'],
    },
    2: {
        'task_name': 'Color-II',
        'task_type': 'color_object',
        'x_space': 'object',
        'theta_space': 'color',
        'x_list': item_dict['object'],
        'theta_list': item_dict['color'],
    },
    3: {
        'task_name': 'Background-I',
        'task_type': 'background_animal',
        'x_space': 'background',
        'theta_space': 'animal',
        'x_list': item_dict['background'],
        'theta_list': item_dict['animal'],
    },
    4: {
        'task_name': 'Background-II',
        'task_type': 'background_animal',
        'x_space': 'animal',
        'theta_space': 'background',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['background'],
    },
    5: {
        'task_name': 'Style-I',
        'task_type': 'style_object',
        'x_space': 'style',
        'theta_space': 'object',
        'x_list': item_dict['style'],
        'theta_list': item_dict['object'],
    },
    6: {
        'task_name': 'Style-II',
        'task_type': 'style_object',
        'x_space': 'object',
        'theta_space': 'style',
        'x_list':  item_dict['object'],
        'theta_list': item_dict['style'],
    },
    7: {
        'task_name': 'Action-I',
        'task_type': 'action_animal',
        'x_space': 'action',
        'theta_space': 'animal',
        'x_list': item_dict['action'],
        'theta_list': item_dict['animal'],
    },
    8: {
        'task_name': 'Action-II',
        'task_type': 'action_animal',
        'x_space': 'animal',
        'theta_space': 'action',
        'x_list': item_dict['animal'],
        'theta_list': item_dict['action'],
    },
    9: {
        'task_name': 'Texture-I',
        'task_type': 'texture_object',
        'x_space': 'texture',
        'theta_space': 'object',
        'x_list': item_dict['texture'],
        'theta_list': item_dict['object'],
    },
    10: {
        'task_name': 'Texture-II',
        'task_type': 'texture_object',
        'x_space': 'object',
        'theta_space': 'texture',
        'x_list': item_dict['object'],
        'theta_list': item_dict['texture'],
    },
}

supported_models = [
    'seed',
]

instruction_dict = {
    'caption': {
        'image': 'We provide a few examples, each with an input, and an output of the image description. Based on the examples, predict the next image description and visualize it. ',
        'text': 'We provide a few examples, each with an input, and an output of the image description. Based on the examples, predict the next image description. ',
    },
    'cot': {
        'general': (
            "Let's analyze the pattern from the given examples.\n\n"
            "Each example contains an object [θ] with varying [x].\n"
            "We'll determine the next step based on the observed pattern.\n",
            "Now, based on the reasoning, let’s describe what comes next.\n"
        ), 
        'image': ('', "\nNow ONLY generate image tokens representing the next object. Wrap them between <img> and </img>. Do not output anything else.\n"), 
        'text': ('', "\nBased on the above, describe what the next image should look like."), 
    },

    'tot': {
        'general': (
            "Now, study  all above examples carefully. What patterns do you notice between the inputs and their corresponding outputs? " 
            "What aspects are changing, and what remains consistent? "
            "Then, considering the new input, think of three different ways the output could be designed. "
            "Be creative but stay logically grounded. Finally, choose the best idea and justify why.\n\n"
            "Thought 1:\n"
        ),
        'image': (
            '',
            "[INST] Generate ONLY image tokens for: <img><Concise description></img> [/INST]"
        ),
        'text': (
            '',
            "[INST] Describe the image in under 15 words [/INST]"
        )
    },

    'instruct': {
        'image': {
            1: 'Please identify the common main object in the images, and generate another image of this object of the requested color. ',
            2: 'Please identify the common color in the images, and generate another image of the requested object in the same color. ',
            3: 'Please identify the common animal in the images, and generate another image of this animal walking in the requested background. ',
            4: 'Please identify the common background in the images, and generate another image of the requested animal walking in the same background. ',
            5: 'Please identify the common object in the images, and generate another image of this object in the requested style. ',
            6: 'Please identify the common style in the images, and generate another image of the requested object in the same style. ',
            7: 'Please identify the common animal in the images, and generate another image of this animal doing the requested action. ',
            8: 'Please identify the common action/mood the animal is doing in the images, and generate another image of the requested animal doing the same action/mood. ',
            9: 'Please identify the common main object in the images, and generate another image of this object of the requested texture. ',
            10: 'Please identify the common texture of the objects in the images, and generate another image of the requested object in the same texture. ',
        },
        'text': {
            1: 'Please identify the common main object in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common main object and the requested color. ',
            2: 'Please identify the common main color in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested object and the common color. ',
            3: 'Please identify the common animal in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common animal and the requested background. ',
            4: 'Please identify the common background in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested animal and the common background. ',
            5: 'Please identify the common object in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common object and the requested style. ',
            6: 'Please identify the common style in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested object and the common style. ',
            7: 'Please identify the common animal in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common animal and the requested action. ',
            8: 'Please identify the common action/mood the animal is doing in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested animal and the common action/mood. ',
            9: 'Please identify the common main object in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the common main object and the requested texture. ',
            10: 'Please identify the common texture of the objects in the images, and describe the next image to be generated based on the sequence below. Your description of image should contain the description of the requested object and the common texture. ',
        },
    },
    'default': {
        'text': {
            'seed': (
                "Please identify the common main object in the images, "
                "and generate another image of this object in the requested color.\n\n"
                "[INST] Example 1: A car that is blue.\n"
                "Example 2: A car that is brown.\n"
                "→ Generate a car that is orange.\n\n"
                "Output only image tokens between <img> and </img>. Do not explain. [/INST]",
                "",
            ),
            
        },
        'image': {
            'seed': (
                 '[INST] Generate ONLY <img>tokens</img> [/INST]'
                 '',
            )
            
        }
    }
}

prompt_type_options = [
    'caption',#  -2, # replace image with image captions
    'instruct', # -1, # tell the prompt to generate the object of the common attribute
    'default', # 0, # basic
    'misleading', # 1, # misleading
    'cot', # 2, # chain of thought
    'exact', # 3, # exact
    'tot', # tree of thought
]

data_modes = [
    'default',
    'ft_train',
    'ft_test',
]

num_prompt_dict = {
    'default': 1000,
    'ft_test': 250,
    'ft_train': 1000,
}