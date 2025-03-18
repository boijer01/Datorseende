import os
import cv2
import numpy as np
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# KOM IHÅG MODELLERNA - de ska finnas placerade utanför skriptet på samma nivå #

def download_images(query, google_max, bing_max, google_folder, bing_folder):
    """
    Laddar ner bilder från Google och Bing
    Bilderna kommer att sparas skilt i olika mappar (google och bing)
    """
    #Starta en crawler för Google och hämta bilder
    google_crawler = GoogleImageCrawler(storage={'root_dir': google_folder})
    google_crawler.crawl(keyword=query, max_num=google_max)

    #Starta en crawler för Bing och hämta bilder (denna var sämre)
    bing_crawler = BingImageCrawler(storage={'root_dir': bing_folder})
    bing_crawler.crawl(keyword=query, max_num=bing_max)

def detect_person(image_path, net, confidence_threshold=0.5):
    """
    Använder MobileNetSSD för att identifiera personer i en bild
    Returnerar bounding box (startX, startY, endX, endY) för den största detekterade personen
    Om ingen person hittas returneras None
    """
    #Läs in bilden
    image = cv2.imread(image_path)
    if image is None:
        print(f"Kunde inte läsa bilden: {image_path}")
        return None

    (h, w) = image.shape[:2]  #Hämtar bildens höjd och bredd

    #Förbered bilden för nätverket (konverterar till blob)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []  #Lista för att lagra identifierade personer

    #Loopa igenom alla detekterade objekt
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  #Hämta konfidensvärdet
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])  #Hämta klass-ID
            if idx == 15:  #Klass-ID 15 motsvarar "person"
                #Skalera bounding box till originalbildens storlek
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))

    #Returnera den största bounding boxen om vi har någon detekterad person
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)

def crop_image(image_path, box, save_path):
    """
    Beskär bilden enligt bounding boxen och sparar den beskurna versionen
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Kunde inte läsa bilden :( {image_path}")
        return

    (startX, startY, endX, endY) = box
    cropped = image[startY:endY, startX:endX]  #Beskär bilden
    cv2.imwrite(save_path, cropped)  #Spara beskuren bild
    print(f"Beskarad bild sparad: {save_path}")

def collect_image_paths(folder):
    """
    Samlar alla bildfiler från en angiven mapp och returnerar deras sökvägar
    """
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    """
    1. Ladda ner bilder från Google och Bing
    2. Identifiera personer i bilderna
    3. Beskära bilderna för att endast visa identifierade personer
    """
    categories = {
        "snowboard": "snowboard downhill"
    }

    #Hur många bilder som ska crawlas
    google_max = 100 
    bing_max = 0  #Inget från Bing atm då vi fick ganska dåligt resultat

    #Mappar för att spara bilder
    base_download_folder = "downloaded_images"
    base_cropped_folder = "cropped_images"

    #Skapa mappar om de inte finns
    os.makedirs(base_download_folder, exist_ok=True)
    os.makedirs(base_cropped_folder, exist_ok=True)

    #Kontrollera att modellfilerna finns
    prototxt = "./MobileNetSSD_deploy.prototxt"
    model = "./MobileNetSSD_deploy.caffemodel"
    
    if not os.path.exists(prototxt) or not os.path.exists(model):
        print("Modellfiler saknas! Ladda ner MobileNetSSD_deploy.prototxt och MobileNetSSD_deploy.caffemodel och placera dem i samma mapp.")
        return
    
    #Ladda in MobileNetSSD-modellen
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    #Ladda ner bilder för varje kategori
    for category, query in categories.items():
        #Skapa specifika mappar för varje kategori
        google_folder = os.path.join(base_download_folder, category, "google")
        bing_folder = os.path.join(base_download_folder, category, "bing")
        os.makedirs(google_folder, exist_ok=True)
        os.makedirs(bing_folder, exist_ok=True)
        
        #Anropa funktionen för att ladda ner bilder
        download_images(query, google_max, bing_max, google_folder, bing_folder)

if __name__ == "__main__":
    main()
