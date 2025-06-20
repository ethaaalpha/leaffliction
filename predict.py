import argparse
import torch
import joblib
from torchvision import transforms as T
from src.model.MiniMobileNet import MiniMobileNet


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
    
    # images_path = # Met le path du csv avec les chemins des images ici
    

   

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
