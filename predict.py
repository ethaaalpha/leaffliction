import argparse
import os
import shutil
import tempfile
import torch
import joblib
import sys
from contextlib import contextmanager
from torchvision import transforms as T

sys.path.append("./src")
from metadata import generate_metadata
from Transformation import generate_transformation, save_histogram_as_image
from preprocessing.loader import Loader
from model.MiniMobileNet import MiniMobileNet

@contextmanager
def get_tmp_images(image_path: str):
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        raise ValueError("Invalid image path!")

    root_dir = tempfile.mkdtemp()
    try:
        _, ext = Loader.get_name_ext(image_path)
        transf_dir = os.path.join(root_dir, "data")
        tmp_image = os.path.join(transf_dir, f"image{ext}")
        histogram_image = os.path.join(root_dir, "color_histogram.jpg")
        csv_file = os.path.join(root_dir, "predict.csv")

        os.makedirs(transf_dir)
        shutil.copyfile(image_path, tmp_image)

        images_transformed = generate_transformation(transf_dir, [tmp_image])
        save_histogram_as_image(tmp_image, histogram_image)
        generate_metadata(transf_dir, csv_file, True)

        yield (csv_file, [*images_transformed, histogram_image])
    finally:
        shutil.rmtree(root_dir)

def predict(model_path, image_path, encoder_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.37272489070892334, 0.3830896019935608, 0.3935730457305908], 
                    std=[0.31965553760528564, 0.3071448802947998, 0.3268057703971863])
        ])

    model = MiniMobileNet(csv_dim=1503, n_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    label_encoder = joblib.load(encoder_path)

    # --- Générer les variantes d’image ---

    # Ici tu peux faire le code qui permet a de Image Path donner en argument cree dans un tmp
    # les images du transform + csv + image de l'histograme (je dois avoir le csv pour le model mais l'image egalementpour l'afficher comme demande le sujet)
    # dans le tmp je veux egalement la csv a une ligne avec "class,original,images" mais avec class vide 

    with get_tmp_images(image_path) as (csv_path, images_path):
        # le csv_path comme demandé et images_path qui contient les 6 transformations sous forme d'image (avec l'histograme)
        # le dossier temporaire est automatiquement supprimé quand tu sors du contexte du with (c'est avec contextmanager et yield)
        print(csv_path)
        print(images_path)

    # --- Prédiction ---
    # with torch.no_grad():
    #     output = model(imgs_tensor, hist_tensor)
    #     pred = output.argmax(dim=1).item()

    # predicted_class = label_encoder.inverse_transform([pred])[0]

    # --- Affichage ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the trained model (.pt)")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--encoder", required=True, help="Path to the label_encoder.pkl")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict(args.model, args.image, args.encoder, device=device)
