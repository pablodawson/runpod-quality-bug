import runpod
import base64
import time

example_id = 2

image = f"examples/room{example_id}.jpg"
mask = f"examples/mask{example_id}.png"

# to base64
with open(image, "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode('utf-8')
with open(mask, "rb") as mask_file:
    mask = base64.b64encode(mask_file.read()).decode('utf-8')

lora = False

if lora:
    model_inputs = {"image_b64": image, "mask_b64": mask, "width": 512, 'use_lora':True}
else:
    model_inputs = {"prompt": "a room with a sofa","image_b64": image, "mask_b64": mask, "width": 512, 'use_lora':False}

runpod.api_key = ""
endpoint = runpod.Endpoint("")

run_request = endpoint.run(model_inputs)

timestart = time.time()

print(run_request.status())
output = run_request.output()

print("Time to run: ", time.time() - timestart)

image = output[0]['image_b64']

# Decode the base64 image
image = base64.b64decode(image.encode('utf-8'))

# Save the image
with open(f"examples/output{example_id}.jpg", "wb") as f:
    f.write(image)