import os
from components.data_loader.data_load import solar_panel_data

os.mkdir("Results-Folder")
with open("./Results-Folder/result.txt", "wb") as file:
    file.write(b"Stefan er nice og det samme er Martin")