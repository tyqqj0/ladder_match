import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def promt_make_option(responce):
    system_prompt = """
      You are a text transformation assistant. Your task is to extract the letter representing the answer (either A, B, C, or D) from a sentence, which typically follows phrases such as "The correct answer is", "The answer is", "Final Answer is", or simply states a response like "D is response". You should only return the letter (A, B, C, or D) and nothing else.

    ###
    Example 1:
    
    Input:  "The correct answer is: **B**."
    Output: "B"
          
    ###
    Example 2:
    
    Input:  "The answer is: **D**."
    Output: "D"
       
    ###
    Example 3: 
    
    Input: 'Final Answer is C.'
    Output: "C"

    ###
    Example 4:

    Input: 'A is response.' 
    Output: "A"
      
    """

    user_prompt = f"""
    Input: {responce}
    Output:
    """
    return system_prompt, user_prompt

def prompt_1_llm(original_shape_coordinates, steps, target_action_str, shape_items):
    system_prompt = """
    You are a player of the game Shapez, and your goal is to transform a set of raw shapes into a target shape through a series of operations.
    First, I'll introduce the game's shape notation:
    Each shape is represented by two characters, where the first character denotes the color, and the second character denotes the shape type. The color and type correspond as follows:
    C: Circle; R: Rectangle; W: Windmill; Fan; S: Star; r: Red; g: Green; b: Blue ; y: Yellow; p: Purple; c: Cyan; u: Uncolored; w: White; --: Empty (No shape or color)
    A single layer's shape code consists of the shape codes for four quadrants, in the following order: 'Quadrant 1, Quadrant 2, Quadrant 3, Quadrant 4'
    For example, 'Su--Ry--' indicates that Quadrant 1 'Su' has an uncolored star, Quadrant 2 '--' is empty, Quadrant 3 'Ry' has a yellow rectangle, and Quadrant 4 '--' is empty.
    A shape can consist of one layers, with each layer represented by a shape code of up to four quadrants.
    Next, I’ll explain the available operations and the rules for each:
    Cutting (removes shapes in Quadrants 1 and 2 of the input shape, which is equal to cut the right side of the shape, such as 'SuSuSuSu' will be cut as '------SuSu');
    Rotate clockwise by 90° (rotates all shapes clockwise by 90°, so Quadrant 1 becomes Quadrant 2, Quadrant 2 becomes Quadrant 3, Quadrant 3 becomes Quadrant 4, and Quadrant 4 becomes Q  uadrant 1), such as 'CgScScCg' will be rotate clockwise as 'CgCgScSc');
    Rotate counterclockwise by 90° (rotates all shapes counterclockwise by 90°, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 3, Quadrant 3 becomes Quadrant 2, and Quadrant 2 becomes Quadrant 1), such as 'CgScScCg' will be rotate counterclockwise as 'ScScCgCg');
    Filling (fill all blank quadrants within the input shape using another shape. Such as fill 'SuSu---—' using 'Rb', the result will be 'SuSuRbRb');
    Mirror (perform a horizontal mirror operation on the entire shape, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 1, Quadrant 2 becomes Quadrant 3, and Quadrant 3 becomes Quadrant 2), such as '----SwSw' will be rotate counterclockwise as 'SwSw----');
    Coloring (input a shape and a color; change the color of every shape in each quadrant of the top layer to the given color, such as colorize {layer_1: 'SuSu---—', layer_2: 'RbRbRbRb'} with color 'p', the result will be {layer_1: 'SuSu---—', layer_2: 'RpRpRpRp'});
    Great, now you have become a Shapez master. From now on, you will answer my questions, and you only need to output the letter corresponding to your choice.
"""

    user_prompt = f"""
    This is an original shape, with the coordinates: {original_shape_coordinates}.
    If you perform {steps} operations on this shape, what will the resulting configuration look like? 
    The operations are:
    {target_action_str}.
    Please select the correct answer from the options below. You only need to output the letter corresponding to your choice.
    The options are:
    A: {shape_items['A']['shape']},
    B: {shape_items['B']['shape']},
    C: {shape_items['C']['shape']},
    D: {shape_items['D']['shape']},
     """

    return system_prompt, user_prompt

def prompt_1_mllm(image_path, steps, target_action_str):
    base64_images = [encode_image(single_image_path) for single_image_path in image_path.values()]
    images = []

    for i, base64_image in enumerate(base64_images):
        images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    system_prompt = """
    You are a player of the game Shapez, and your goal is to transform a set of raw shapes into a target shape through a series of operations.
    First, I'll introduce the game's rules:
    A shape can consist of one layers, with each layer represented by a shape code of up to four quadrants. The upper-right area represents the Quadrant 1, the lower-right the Quadrant 2, the lower-left the Quadrant 3, and the upper-left the Quadrant 4. 
    Next, I’ll explain the available operations and the rules for each:
    Cutting (removes shapes in Quadrants 1 and 2 of the input shape, which is equal to cut the right side of the shape);
    Rotate clockwise by 90° (rotates all shapes clockwise by 90°, so Quadrant 1 becomes Quadrant 2, Quadrant 2 becomes Quadrant 3, Quadrant 3 becomes Quadrant 4, and Quadrant 4 becomes Q  uadrant 1);
    Rotate counterclockwise by 90° (rotates all shapes counterclockwise by 90°, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 3, Quadrant 3 becomes Quadrant 2, and Quadrant 2 becomes Quadrant 1);
    Filling (fill all blank quadrants within the input shape using another shape. Such as fill 'SuSu---—' using 'Rb', the result will be 'SuSuRbRb');
    Mirror (perform a horizontal mirror operation on the entire shape, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 1, Quadrant 2 becomes Quadrant 3, and Quadrant 3 becomes Quadrant 2);
    Coloring (input a shape and a color; change the color of every shape in each quadrant of the top layer to the given color);
    Great, now you have become a Shapez master. From now on, you will answer my questions, and you only need to output the letter corresponding to your choice.
"""

    user_prompt = [
    {
        "type": "text",
        "text": "This is the original shape, with its image:."
    },
    images[0],
    {
        "type": "text",
        "text": f"If you perform {steps} operations on this shape, what will the resulting configuration look like?  The operations are: {target_action_str}. Please select the correct answer from the options below. You only need to output the letter corresponding to your choice. The options are: "
    },
    {
        "type": "text",
        "text": "A: "
    },
    images[1],
    {
        "type": "text",
        "text": "B: "
    },
    images[2],
    {
        "type": "text",
        "text": "C: "
    },
    images[3],
    {
        "type": "text",
        "text": "D: "
    },
    images[4],
]
    
    return system_prompt, user_prompt

def prompt_2_llm(original_shape_coordinates, steps, target_shape_coordinates, shape_items):
    system_prompt = """
    You are a player of the game Shapez, and your goal is to transform a set of raw shapes into a target shape through a series of operations.
    First, I'll introduce the game's shape notation:
    Each shape is represented by two characters, where the first character denotes the color, and the second character denotes the shape type. The color and type correspond as follows:
    C: Circle; R: Rectangle; W: Windmill; Fan; S: Star; r: Red; g: Green; b: Blue ; y: Yellow; p: Purple; c: Cyan; u: Uncolored; w: White; --: Empty (No shape or color)
    A single layer's shape code consists of the shape codes for four quadrants, in the following order: 'Quadrant 1, Quadrant 2, Quadrant 3, Quadrant 4'
    For example, 'Su--Ry--' indicates that Quadrant 1 'Su' has an uncolored star, Quadrant 2 '--' is empty, Quadrant 3 'Ry' has a yellow rectangle, and Quadrant 4 '--' is empty.
    A shape can consist of one layers, with each layer represented by a shape code of up to four quadrants.
    Next, I’ll explain the available operations and the rules for each:
    Cutting (removes shapes in Quadrants 1 and 2 of the input shape, which is equal to cut the right side of the shape, such as 'SuSuSuSu' will be cut as '------SuSu');
    Rotate clockwise by 90° (rotates all shapes clockwise by 90°, so Quadrant 1 becomes Quadrant 2, Quadrant 2 becomes Quadrant 3, Quadrant 3 becomes Quadrant 4, and Quadrant 4 becomes Q  uadrant 1), such as 'CgScScCg' will be rotate clockwise as 'CgCgScSc');
    Rotate counterclockwise by 90° (rotates all shapes counterclockwise by 90°, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 3, Quadrant 3 becomes Quadrant 2, and Quadrant 2 becomes Quadrant 1), such as 'CgScScCg' will be rotate counterclockwise as 'ScScCgCg');
    Filling (fill all blank quadrants within the input shape using another shape. Such as fill 'SuSu---—' using 'Rb', the result will be 'SuSuRbRb');
    Mirror (perform a horizontal mirror operation on the entire shape, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 1, Quadrant 2 becomes Quadrant 3, and Quadrant 3 becomes Quadrant 2), such as '----SwSw' will be rotate counterclockwise as 'SwSw----');
    Coloring (input a shape and a color; change the color of every shape in each quadrant of the top layer to the given color, such as colorize {layer_1: 'SuSu---—', layer_2: 'RbRbRbRb'} with color 'p', the result will be {layer_1: 'SuSu---—', layer_2: 'RpRpRpRp'});
    Great, now you have become a Shapez master. From now on, you will answer my questions, and you only need to output the letter corresponding to your choice.
"""
    user_prompt = f"""
    # This is an original shape, with the coordinates: {original_shape_coordinates}.
    # If this shape is performed with {steps} operations, and the result will be given below, can you infer what the operations are?
    # The result is: {target_shape_coordinates}.
    # Please select the correct answer from the options below. You only need to output the letter corresponding to your choice.
    # The options are:
    # A: {shape_items['A']['action_list']},
    # B: {shape_items['B']['action_list']},
    # C: {shape_items['C']['action_list']},
    # D: {shape_items['D']['action_list']}
    """

    return system_prompt, user_prompt

def prompt_2_mllm(original_image_path, correct_option_image_path, steps, shape_items):
    base64_original_image = encode_image(original_image_path)
    base64_correct_option_image = encode_image(correct_option_image_path)
    images = []

    images.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_original_image}"
        }
    })
    images.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_correct_option_image}"
        }
    })
    
    system_prompt = """
    You are a player of the game Shapez, and your goal is to transform a set of raw shapes into a target shape through a series of operations.
    First, I'll introduce the game's rules:
    A shape can consist of one layers, with each layer represented by a shape code of up to four quadrants. The upper-right area represents the Quadrant 1, the lower-right the Quadrant 2, the lower-left the Quadrant 3, and the upper-left the Quadrant 4. 
    Next, I’ll explain the available operations and the rules for each:
    Cutting (removes shapes in Quadrants 1 and 2 of the input shape, which is equal to cut the right side of the shape);
    Rotate clockwise by 90° (rotates all shapes clockwise by 90°, so Quadrant 1 becomes Quadrant 2, Quadrant 2 becomes Quadrant 3, Quadrant 3 becomes Quadrant 4, and Quadrant 4 becomes Q  uadrant 1);
    Rotate counterclockwise by 90° (rotates all shapes counterclockwise by 90°, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 3, Quadrant 3 becomes Quadrant 2, and Quadrant 2 becomes Quadrant 1);
    Filling (fill all blank quadrants within the input shape using another shape. Such as fill 'SuSu---—' using 'Rb', the result will be 'SuSuRbRb');
    Mirror (perform a horizontal mirror operation on the entire shape, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 1, Quadrant 2 becomes Quadrant 3, and Quadrant 3 becomes Quadrant 2);
    Coloring (input a shape and a color; change the color of every shape in each quadrant of the top layer to the given color);
    Great, now you have become a Shapez master. From now on, you will answer my questions, and you only need to output the letter corresponding to your choice.
"""

    user_prompt = [
    {
        "type": "text",
        "text": "This is an original shape, with itsimage:"
    },
    images[0],
    {
        "type": "text",
        "text": f" If this shape is performed with {steps} operations, and the result will be given below, can you infer what the operations are?The result is: "
    },
    images[1],
    {
        "type": "text",
        "text": f"""
    Please select the correct answer from the options below. You only need to output the letter corresponding to your choice. The options are:
    A: {shape_items['A']['action_list']},
    B: {shape_items['B']['action_list']},
    C: {shape_items['C']['action_list']},
    D: {shape_items['D']['action_list']}
    """
    }
]
    
    return system_prompt, user_prompt



"""
    You are a player of the game Shapez, and your goal is to transform a set of raw shapes into a target shape through a series of operations.
    First, I'll introduce the game's shape notation:
    Each shape is represented by two characters, where the first character denotes the color, and the second character denotes the shape type. The color and type correspond as follows:
    C: Circle; R: Rectangle; W: Windmill; Fan; S: Star; r: Red; g: Green; b: Blue ; y: Yellow; p: Purple; c: Cyan; u: Uncolored; w: White; --: Empty (No shape or color)
    A single layer's shape code consists of the shape codes for four quadrants, in the following order: 'Quadrant 1, Quadrant 2, Quadrant 3, Quadrant 4'
    For example, 'Su--Ry--' indicates that Quadrant 1 'Su' has an uncolored star, Quadrant 2 '--' is empty, Quadrant 3 'Ry' has a yellow rectangle, and Quadrant 4 '--' is empty.
    A shape can consist of multiple layers, with each layer represented by a shape code of up to four quadrants. The layers are arranged from bottom to top, as follows:
    {'Layer 1': 'Shape code for Layer 1', 'Layer 2': 'Shape code for Layer 2', 'Layer 3': 'Shape code for Layer 3', 'Layer 4': 'Shape code for Layer 4'}, the layer with the larger number is on the top.
    Note that not every shape has a fixed number of layers; some shapes have only one layer, while others may have two, three, or four layers.
    The whole shapes follows specific physical rules:
    1.If a layer contains only one quadrant with a shape, that quadrant must have a corresponding shape in the layer below. For example, in {'Layer 1': 'Su--Ry--', 'Layer 2': '------Wp--'}, quadrant 3's 'Wp' is the only shape in Layer 2, so quadrant 3 of the layer below (Layer 1) must also contain a shape, which in this case is 'Ry'. Conversely, in {'Layer 1': 'SuRy----', 'Layer 2': '------Wp--'}, quadrant 3's 'Wp' in Layer 2 is not supported by a shape in the corresponding quadrant of Layer 1, which is blank ('--'). This does not comply with physical laws.
    2.If a layer contains two non-adjacent quadrants with shapes, those two quadrants must also have corresponding shapes in the layer below. For example, in {'Layer 1': 'Su--Ry--', 'Layer 2': 'Sb--Wp--'}, quadrants 1 ('Sb') and 3 ('Wp') are non-adjacent shapes in Layer 2. Therefore, quadrants 1 and 3 in Layer 1 must also contain shapes, as seen in 'Su' and 'Ry'. Conversely, in {'Layer 1': '----Ry--', 'Layer 2': 'Sb--Wp--'}, quadrant 1 in Layer 1 is blank ('--'), which does not conform to physical laws.
    3.If a layer contains two adjacent quadrants with shapes, then at least one of those quadrants must contain a corresponding shape in the layer below. For example, in {'Layer 1': 'Su------', 'Layer 2': 'SbWp----'}, quadrants 1 ('Sb') and 2 ('Wp') are adjacent shapes in Layer 2. Therefore, either quadrant 1 or quadrant 2 in Layer 1 must contain a shape, as shown by 'Su' in quadrant 1. Conversely, in {'Layer 1': '----Ry--', 'Layer 2': 'SbWp----'}, both quadrants 1 and 2 in Layer 1 are blank ('--'), which does not comply with physical laws.
    4.If a layer contains three quadrants with shapes, then at least one of those quadrants must have a corresponding shape in the layer below. For example, in {'Layer 1': '--Su----', 'Layer 2': 'SbWpCb--'}, quadrants 1 ('Sb'), 2 ('Wp'), and 3 ('Cb') are the three shapes in Layer 2. Therefore, at least one of quadrants 1, 2, or 3 in Layer 1 must contain a shape, as seen in quadrant 2 with 'Su'. Conversely, in {'Layer 1': '------Ry', 'Layer 2': 'SbWpCb--'}, all quadrants in Layer 1 are blank ('--'), which does not comply with physical laws.
    Next, I’ll explain the available operations and the rules for each, please remember that all shapes obtained after operations must follow the laws of physics.:
    Cutting (removes shapes in Quadrants 1 and 2 from each layer of the input shape, as well as cut the right side of the shape);
    Rotate clockwise by 90° (rotates all shapes clockwise by 90°, so Quadrant 1 becomes Quadrant 2, Quadrant 2 becomes Quadrant 3, Quadrant 3 becomes Quadrant 4, and Quadrant 4 becomes Quadrant 1);
    Rotate counterclockwise by 90° (rotates all shapes counterclockwise by 90°, so Quadrant 1 becomes Quadrant 4, Quadrant 4 becomes Quadrant 3, Quadrant 3 becomes Quadrant 2, and Quadrant 2 becomes Quadrant 1);
    Stacking (stacks one shape on top of another.).
    Coloring (input a shape and a color; change the color of every shape in each quadrant of the top layer to the given color).
    Next, I will give you a primitive figure and a target figure, and there are 4 operation options. Please choose the correct operation from these four options so that it can make the original figure a target graphic. You only need to answer a letter, don't explain more.
    
    Example:
    original_shape : {'layer1': '--CwRpSb', 'layer2': 'Sc----Su', 'layer3': 'Cr----Ru', 'layer4': '--SuWcCb'}
    target_shape : {'layer1': '----RpSb', 'layer2': '------Su', 'layer3': '------Ru', 'layer4': '----WcCb'}
    options :["A: ['cut']", "B: ['cut']", "C: ['rotate_counterclockwise']", "D: [['colorize', 'y']]"]
    Output: A
"""