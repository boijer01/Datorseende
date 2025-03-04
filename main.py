import os
import cv2
import numpy as np
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

def download_images(query, google_max, bing_max, google_folder, bing_folder):
    """
    Använder icrawler för att hämta bilder från Google och Bing för ett givet sökord.
    Bilderna sparas i mapparna google_folder respektive bing_folder.
    """
    google_crawler = GoogleImageCrawler(storage={'root_dir': google_folder})
    google_crawler.crawl(keyword=query, max_num=google_max)

    bing_crawler = BingImageCrawler(storage={'root_dir': bing_folder})
    bing_crawler.crawl(keyword=query, max_num=bing_max)

def detect_person(image_path, net, confidence_threshold=0.5):
    """
    Använder MobileNetSSD för att detektera personer i bilden.
    Returnerar den största bounding boxen (startX, startY, endX, endY)
    om en person hittas, annars None.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Kunde inte läsa bilden: {image_path}")
        return None
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))

    if boxes:
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[0]
    else:
        return None

def crop_image(image_path, box, save_path):
    """
    Beskärmer bilden enligt bounding boxen och sparar den beskurna bilden.
    """
    image = cv2.imread(image_path)
    (startX, startY, endX, endY) = box
    cropped = image[startY:endY, startX:endX]
    cv2.imwrite(save_path, cropped)
    print(f"Beskarad bild sparad: {save_path}")

def collect_image_paths(folder):
    """
    Går igenom en mapp och returnerar sökvägar till alla bilder med filändelser .jpg, .jpeg och .png.
    """
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    categories = {
        "snowboard": "snowboard downhill"
    }

    google_max = 100 
    bing_max = 0

    base_download_folder = "downloaded_images"
    base_cropped_folder = "cropped_images"


    os.makedirs(base_download_folder, exist_ok=True)
    os.makedirs(base_cropped_folder, exist_ok=True)

    prototxt = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"
    if not os.path.exists(prototxt) or not os.path.exists(model):
        print("Modellfiler saknas! Ladda ner MobileNetSSD_deploy.prototxt och MobileNetSSD_deploy.caffemodel och spara dem i samma mapp.")
        return
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    for category, query in categories.items():
        print(f"\n=== Behandlar kategori: {category} ({query}) ===")

        google_folder = os.path.join(base_download_folder, category, "google")
        bing_folder = os.path.join(base_download_folder, category, "bing")
        os.makedirs(google_folder, exist_ok=True)
        os.makedirs(bing_folder, exist_ok=True)

        print(f"Hämtar bilder för '{query}' från Google och Bing...")
        download_images(query, google_max, bing_max, google_folder, bing_folder)

        category_download_folder = os.path.join(base_download_folder, category)
        image_paths = collect_image_paths(category_download_folder)
        print(f"Totalt nedladdade bilder för {category}: {len(image_paths)}")

        cropped_category_folder = os.path.join(base_cropped_folder, category)
        os.makedirs(cropped_category_folder, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            print(f"\nBearbetar bild {i+1} för {category}: {image_path}")
            box = detect_person(image_path, net)
            if box:
                cropped_filename = f"cropped_{i}.jpg"
                cropped_path = os.path.join(cropped_category_folder, cropped_filename)
                crop_image(image_path, box, cropped_path)
            else:
                print("Ingen person detekterades i bilden.")

main()
