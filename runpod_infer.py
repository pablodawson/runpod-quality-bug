''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate
import json
import base64
from io import BytesIO
from PIL import Image

prod = False

MODEL = predict.Predictor()
MODEL.setup()

INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
        'default': "A room with a rosjf sofa"
    },
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'image_b64': {
        'type': str,
        'required': True,
        'default': None
    },
    'mask_b64': {
        'type': str,
        'required': True,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 20
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': "K-LMS",
        'constraints': lambda scheduler: scheduler in ["DDIM", "DDPM", "DPM-M", "DPM-S", "EULER-A", "EULER-D",
                                                         "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM", "K-LMS"]
    },
    'use_lora': {
        'type': bool,
        'required': False,
        'default': False
    }
}


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # b64 -> Image
    image_bytes = base64.b64decode(validated_input['image_b64'].encode('utf-8'))
    mask_bytes = base64.b64decode(validated_input['mask_b64'].encode('utf-8'))

    if not os.path.exists("input_objects"):
        os.mkdir("input_objects")
    
    image = Image.open(BytesIO(image_bytes)).save('input_objects/image.png')
    mask = Image.open(BytesIO(mask_bytes)).save('input_objects/mask.png')

    img_paths = MODEL.predict(
        prompt=validated_input.get('prompt', "A room with a rosjf sofa"),
        width=validated_input.get('width', 512),
        image= "input_objects/image.png",
        mask= "input_objects/mask.png",
        num_inference_steps=validated_input.get('num_inference_steps', 50),
        guidance_scale=validated_input['guidance_scale'],
        scheduler=validated_input.get('scheduler', "K-LMS"),
        use_lora=validated_input.get('use_lora', False)
    )

    job_output = []

    for path in img_paths:

        buffered = BytesIO()
        Image.open(path).save(buffered, format="JPEG")
        output = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        job_output.append({
            "image_b64": output
        })
    
    # Remove downloaded input objects
    if prod:
        rp_cleanup.clean(['input_objects', 'tmp'])
    
    return job_output

if prod:
    runpod.serverless.start({"handler": run})
else:
    job = {}
    job['id'] = 'test'

    example_id = 0

    image = f"examples/room{example_id}.jpg"
    seg = f"examples/mask{example_id}.png"

    # to base64
    with open(image, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode('utf-8')
    with open(seg, "rb") as seg_file:
        seg = base64.b64encode(seg_file.read()).decode('utf-8')

    job['input'] = { "image_b64": image, "mask_b64": seg, "width": 512, 'use_lora':True}
    
    run(job)