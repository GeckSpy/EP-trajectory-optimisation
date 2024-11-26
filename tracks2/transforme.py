import os
from PIL import Image

# Charger l'image
input_folder = "preimages"
output_folder = "post_images"

def process_images_in_folder(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir tous les fichiers dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):  # Filtrer uniquement les fichiers PNG
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # Ouvrir l'image
            image = Image.open(input_path)

            # Convertir en mode RGB si ce n'est pas déjà le cas
            image = image.convert("RGB")

            # Récupérer les pixels
            pixels = image.load()

            # Traiter chaque pixel
            val_max = 0
            inds = 0,0
            for x in range(image.width):
                for y in range(image.height):
                    r, g, b = pixels[x, y]
                    if r+g+b > val_max :
                        val_max = r+g+b
                        inds = x,y
                    # Échanger le rouge et le bleu
            for x in range(image.width):
                for y in range(image.height):
                    r, g, b = pixels[x,y]
                    if (x,y) != inds :
                        if r > 0 :
                            r -= 1
                        if g > 0 :
                            g -= 1
                        if b > 0 :
                            b -= 1
                    pixels[x,y] = r,g,b
            # Sauvegarder l'image modifiée
            image.save(output_path)

            print(f"Image modifiée sauvegardée à : {output_path}")


process_images_in_folder(input_folder, output_folder)