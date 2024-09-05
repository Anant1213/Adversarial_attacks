from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import numpy as np
import io
from fastapi.responses import StreamingResponse
from attacks.fgsm import fgsm_attack
from model import load_model

app = FastAPI()

# Initialize the model
model = load_model()

@app.post("/attack")
async def attack(image: UploadFile = File(...), attack_type: str = Form(...), epsilon: float = Form(...)):
    # Read the uploaded image
    img = Image.open(io.BytesIO(await image.read()))
    
    # Preprocess the image
    img_array = np.array(img.resize((224, 224))) / 255.0  # Rescale to [0,1]

    # Perform the attack
    if attack_type == 'fgsm':
        adversarial_array = fgsm_attack(model, img_array, epsilon)
    else:
        return {"error": "Invalid attack type"}
    
    # Convert numpy array back to image
    adversarial_image = (adversarial_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(adversarial_image)
    
    # Save image to a bytes buffer
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)  # Move the cursor to the start of the stream
    
    # Return the image as a response using StreamingResponse
    return StreamingResponse(img_buffer, media_type="image/png")
