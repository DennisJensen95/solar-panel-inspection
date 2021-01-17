from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
<<<<<<< HEAD
from components.predictions.predict_faults import predict_faults
=======
from dog_prediction import predict_dogs
>>>>>>> 037097c021a5de17c6216d64006c7388e1be5882
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
<<<<<<< HEAD
    model_string = "./Results-folder/solar_model_mask_fault-classification_20210114-093235/solar_model_mask_fault-classification_20210114-093235"
    predict_faults(img, model_string)
=======
    predict_dogs(file_path)
>>>>>>> 037097c021a5de17c6216d64006c7388e1be5882

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
