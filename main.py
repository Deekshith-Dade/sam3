import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image_path="test_img.jpg"
image = Image.open(image_path)
inference_state = processor.set_image(image)
# Prompt the model with text
prompt = "lemons"
output = processor.set_text_prompt(
   state=inference_state, prompt=prompt
)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
breakpoint()
