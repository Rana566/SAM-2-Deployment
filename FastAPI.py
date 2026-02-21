from fastapi import FastAPI, UploadFile, File,Form
import uvicorn
import os
import cv2
from typing import List
import numpy as np
from fastapi.responses import StreamingResponse
import json 
import io


app=FastAPI()

@app.get("/")
def print_welcome():
    return {"welcome first fastapi get method"}


def process_image(img_np: np.ndarray, coords: dict) -> np.ndarray:
    """
    img_np: image as numpy array
    coords: {"x": int, "y": int}
    return: image as numpy array
    """
    x, y = coords["x"], coords["y"]

    output = img_np.copy()
    cv2.circle(output, (x, y), 40, (0, 255, 0), 3)

    return output
    
@app.post("/img_sam")
def img_sam(image: UploadFile = File(...),
    coordinates: List=[]):
    
    contents = image.read()
    img_np = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )
    
    #fun process img 
    img=process_image(contents,coordinates)
    coords = json.loads(coordinates)
    result_img = process_image(img_np, coords)

    _, buffer = cv2.imencode(".png", result_img)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")
     
if __name__ == "__main__":
    uvicorn.run(app)    