from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from dog_prediction import predict_dogs
import io
import numpy as np
import cv2
import os

app = FastAPI()

origins = ["https://localhost:3000", "http://localhost:3000", "localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def name_image_and_save_it(file, img_obj):
    # Remove previous images and
    os.system("rm media/analyze_image*")
    filename, extension = os.path.splitext(file.filename)
    file_path = "media/analyze_image" + extension
    cv2.imwrite(file_path, img_obj)

    return file_path, extension


@app.post("/image")
async def image(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    file_path, extension = name_image_and_save_it(image, img)

    img_dimensions = str(img.shape)
    predict_dogs(file_path)

    # line that fixed it
    _, encoded_img = cv2.imencode('.PNG', img)
    media_type = "image/" + extension
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type=extension)


# @app.post("/image")
# async def image(image: UploadFile = File(...)):
#     filename, extension = os.path.splitext(image.filename)
#     os.system("rm media/analyze_image*")
#     file_path = "media/analyze_image" + extension
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(image.file, buffer)
#     predict_dogs(file_path)
#     return {"filename": image.filename, "image": image}
