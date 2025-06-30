import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
from pathlib import Path
import cv2
import numpy as np
import re
from Levenshtein import distance as levenshtein_distance  
from tqdm import tqdm


def load_model(method_name, model_path):
    print(f"Loading model for {method_name} from {model_path}...")
    return None 

def evaluate(method_name, split="real", model_path=None, output_path='./results'):

    import os
    os.makedirs(output_path, exist_ok=True)

    # DATASET SELECTION 
    if split=="real":
        print("Evaluating on real images...")
        dataset_path = r"C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Data\Test"
        image_files = list(Path(dataset_path).rglob("*.bmp"))

        json_data_path = r"C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\Evaluation\Annotations\real_annotations.json"
        with open(json_data_path, 'r') as f:
            annotations = json.load(f)
       
    elif split=="synthetic-sd":
        print("Evaluating on Stable Diffusion - synthetic images ...")

    elif split=="synthetic-s":
        print("Evaluating on synthetic images ...")

    else:
        raise ValueError("Invalid split. Choose from 'real', 'synthetic-sd', or 'synthetic-s'.")
        return
    
    # METHOD SELECTION
    if method_name == "roinet_tesseract":

        import pytesseract
        from PIL import Image
        import time
        import cv2
        from tqdm import tqdm
        import sys
        import re
        sys.path.append(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\RoiNet')
        import inference


        print("Using Tesseract OCR for evaluation...")

        model = inference.load_model(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\RoiNet\model_ir.pth')

        total_time = 0
        total_images = len(image_files)
        results = []

        start_time = time.time()

        for img_path in tqdm(image_files, desc="Evaluating Images", leave=False):

            # print(f"Processing image: {img_path}")
          
            image, points = inference.predict_points(model, img_path)
            image_with_points = inference.draw_points_and_polygon(image, points)
            cropped_img = inference.crop_image_smaller(image, points, 0.02)

            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            eroded = cv2.erode(blurred, None, iterations=1)
            dilated = cv2.dilate(eroded, None, iterations=1)

            binary = cv2.adaptiveThreshold(
                dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            inverted = cv2.bitwise_not(binary)
            processed_image = Image.fromarray(inverted)

            ocr_result = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            raw_text = " ".join(ocr_result['text']).strip().replace("\n", " ")
            ocr_text = re.sub(r'\s+', ' ', raw_text)

            real_code = annotations.get(img_path.name, {}).get("text", "").replace("\n", " ")

            real_code = real_code.replace(" ", "")
            ocr_text = ocr_text.replace(" ", "")

            lev_distance = levenshtein_distance(ocr_text, real_code)
            # print(f"Levenshtein distance: {lev_distance}")
        
            if len(real_code) > 0:
                cer = lev_distance / len(real_code)
            else:
                cer = 1.0 

            exact_match = ocr_text == real_code
            # print(f"Exact match: {exact_match}")

            result = {
                "image_path": str(img_path),
                "ocr_text": ocr_text,
                "real_code": real_code,
                "levenshtein_distance": lev_distance,
                "cer": cer,
                "exact_match": exact_match
            }
            results.append(result)
        
        total_time += time.time() - start_time
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
        print(f"Average CER: {sum(r['cer'] for r in results) / total_images * 100:.2f}%")
        print(f"Exact match rate: {sum(r['exact_match'] for r in results) / total_images * 100:.2f}%")
        print(f"Total images processed: {total_images}")

        # Append total time and average CER to results
        results.append({
            "total_time": total_time,
            "average_time_per_image": total_time / total_images,
            "average_cer": sum(r['cer'] for r in results) / total_images * 100,
            "exact_match_rate": sum(r['exact_match'] for r in results) / total_images * 100,
            "total_images": total_images
        })

        # Save results to JSON
        output_file = os.path.join(output_path, f"{method_name}_evaluation_results_split_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)    

    elif method_name == "roinet_easyocr":

        print("Using EasyOCR for evaluation...")

        import easyocr
        import time
        import re
        from tqdm import tqdm
        reader = easyocr.Reader(['en', 'es'])
        import sys
        sys.path.append(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\RoiNet')
        import inference

        model = inference.load_model(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\RoiNet\model_ir.pth')

        total_time = 0
        total_images = len(image_files)
        results = []

        start_time = time.time()

        for img_path in tqdm(image_files, desc="Evaluating Images", leave=False):

            image, points = inference.predict_points(model, img_path)
            cropped_img = inference.crop_image_smaller(image, points, 0.02)

            result = reader.readtext(cropped_img)
            ocr_text = " ".join([text.replace(" ", "") for (_, text, _) in result])

            real_code = annotations.get(img_path.name, {}).get("text", "").replace("\n", " ")
            
            real_code = real_code.replace(" ", "")
            ocr_text = ocr_text.replace(" ", "")

            lev_distance = levenshtein_distance(ocr_text, real_code)
            # print(f"Levenshtein distance: {lev_distance}")
        
            if len(real_code) > 0:
                cer = lev_distance / len(real_code)
            else:
                cer = 1.0 

            exact_match = ocr_text == real_code
            # print(f"Exact match: {exact_match}")

            result = {
                "image_path": str(img_path),
                "ocr_text": ocr_text,
                "real_code": real_code,
                "levenshtein_distance": lev_distance,
                "cer": cer,
                "exact_match": exact_match
            }
            results.append(result)
        
        total_time += time.time() - start_time
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
        print(f"Average CER: {sum(r['cer'] for r in results) / total_images * 100:.2f}%")
        print(f"Exact match rate: {sum(r['exact_match'] for r in results) / total_images * 100:.2f}%")
        print(f"Total images processed: {total_images}")

        results.append({
            "total_time": total_time,
            "average_time_per_image": total_time / total_images,
            "average_cer": sum(r['cer'] for r in results) / total_images * 100,
            "exact_match_rate": sum(r['exact_match'] for r in results) / total_images * 100,
            "total_images": total_images
        })

        # Save results to JSON
        output_file = os.path.join(output_path, f"{method_name}_evaluation_results_split_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)   

            
    elif method_name == "roinet_paddle":

        print("Using PaddleOCR for evaluation...")

        from paddleocr import PaddleOCR, draw_ocr
        ocr = PaddleOCR(use_angle_cls = True, show_log=False)
        import time
        import re
        from tqdm import tqdm
        import os
        import sys
        sys.path.append(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\RoiNet')
        import inference

        model = inference.load_model(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\RoiNet\model_ir.pth')

        total_time = 0
        total_images = len(image_files)
        results = []

        start_time = time.time()

        for img_path in tqdm(image_files, desc="Evaluating Images", leave=False):

            image, points = inference.predict_points(model, img_path) # this is the image with the points drawn
            cropped_img = inference.crop_image_smaller(image, points, 0.02) # 

            result = ocr.ocr(cropped_img, cls=True)
            ocr_text = " ".join([line[1][0] for sublist in result for line in sublist])
            ocr_text = re.sub(r'\s+', ' ', ocr_text).strip() 
           
            real_code = annotations.get(img_path.name, {}).get("text", "").replace("\n", " ")

            real_code = re.sub(r'\s+', ' ', real_code).strip()  # Normalize spaces in real code
            ocr_text = ocr_text.replace(" ", "") 

            real_code = real_code.replace(" ", "")
            ocr_text = ocr_text.replace(" ", "")

            lev_distance = levenshtein_distance(ocr_text, real_code)
            # print(f"Levenshtein distance: {lev_distance}")
        
            if len(real_code) > 0:
                cer = lev_distance / len(real_code)
            else:
                cer = 1.0 

            exact_match = ocr_text == real_code
            # print(f"Exact match: {exact_match}")

            result = {
                "image_path": str(img_path),
                "ocr_text": ocr_text,
                "real_code": real_code,
                "levenshtein_distance": lev_distance,
                "cer": cer,
                "exact_match": exact_match
            }
            results.append(result)
        
        total_time += time.time() - start_time
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
        print(f"Average CER: {sum(r['cer'] for r in results) / total_images * 100:.2f}%")
        print(f"Exact match rate: {sum(r['exact_match'] for r in results) / total_images * 100:.2f}%")
        print(f"Total images processed: {total_images}")

        # Append total time and average CER to results
        results.append({
            "total_time": total_time,
            "average_time_per_image": total_time / total_images,
            "average_cer": sum(r['cer'] for r in results) / total_images * 100,
            "exact_match_rate": sum(r['exact_match'] for r in results) / total_images * 100,
            "total_images": total_images
        })


        # Save results to JSON
        output_file = os.path.join(output_path, f"{method_name}_evaluation_results_split_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)   

    elif method_name == "unet_paddle":

        import os
        os.environ["SM_FRAMEWORK"] = "tf.keras"

        import efficientnet.tfkeras  # this replaces the old efficientnet

        import segmentation_models as sm
        sm.set_framework('tf.keras')
        sm.framework()

        from segmentation_models import Unet
        from segmentation_models.losses import DiceLoss
        from segmentation_models.metrics import iou_score
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
        import cv2
        import numpy as np
        from paddleocr import PaddleOCR
        import time
        import re

        def get_bounding_boxes(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))

            return boxes

        input_shape = (256, 256, 3)

        model = Unet(
            backbone_name='resnet34',
            input_shape=input_shape,
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid'
        )

        model.load_weights(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\Unet\unet_model.keras')

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=DiceLoss(),
            metrics=[iou_score]
        )

        ocr = PaddleOCR(use_angle_cls=True)
        input_size = (256, 256)

        total_time = 0
        total_images = len(image_files)  # Make sure image_files is defined in your script
        results = []

        start_time = time.time()

        for img_path in image_files:
            original_image = cv2.imread(str(img_path))
            image = original_image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, input_size)
            image_resized = image_resized.astype(np.float32) / 255.0
            input_img = np.expand_dims(image_resized, axis=0)

            predicted_mask = model.predict(input_img)
            predicted_mask = (predicted_mask > 0.4).astype(np.uint8)
            predicted_mask = predicted_mask[0, :, :, 0]
            output_size = (800, 600)
            predicted_mask_upscaled = cv2.resize(predicted_mask, output_size, interpolation=cv2.INTER_NEAREST)

            bboxes = get_bounding_boxes(predicted_mask_upscaled)

            original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            original_image_eq = cv2.equalizeHist(original_image_gray)
            original_image = cv2.cvtColor(original_image_eq, cv2.COLOR_GRAY2BGR)

            masked_image = np.ones_like(original_image) * 255

            for bbox in bboxes:
                x, y, w, h = bbox
                masked_image[y:y+h, x:x+w] = original_image[y:y+h, x:x+w]

            results_ocr = ocr.ocr(masked_image)

            img_code = ""

            for result in results_ocr:
                for line in result:
                    box = line[0]
                    text = line[1][0]
                    score = line[1][1]
                    img_code += text + " "

            real_code = annotations.get(img_path.name, {}).get("text", "").replace("\n", " ")

            real_code = real_code.replace(" ", "")
            img_code = img_code.replace(" ", "")

            lev_distance = levenshtein_distance(img_code.strip(), real_code)
            cer = lev_distance / len(real_code) if len(real_code) > 0 else 1.0
            exact_match = img_code.strip() == real_code

            results.append({
                "image_path": str(img_path),
                "ocr_text": img_code.strip(),
                "real_code": real_code,
                "levenshtein_distance": lev_distance,
                "cer": cer,
                "exact_match": exact_match
            })

        total_time += time.time() - start_time
        
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
        print(f"Average CER: {sum(r['cer'] for r in results) / total_images * 100:.2f}%")
        print(f"Exact match rate: {sum(r['exact_match'] for r in results) / total_images * 100:.2f}%")
        print(f"Total images processed: {total_images}")

        # Append total time and average CER to results
        results.append({
            "total_time": total_time,
            "average_time_per_image": total_time / total_images,
            "average_cer": sum(r['cer'] for r in results) / total_images * 100,
            "exact_match_rate": sum(r['exact_match'] for r in results) / total_images * 100,
            "total_images": total_images
        })


        # Save results to JSON
        output_file = os.path.join(output_path, f"{method_name}_evaluation_results_split_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)   

    elif method_name == "unet_paddle_crop":

        import os
        os.environ["SM_FRAMEWORK"] = "tf.keras"

        import efficientnet.tfkeras  # this replaces the old efficientnet

        import segmentation_models as sm
        sm.set_framework('tf.keras')
        sm.framework()

        from segmentation_models import Unet
        from segmentation_models.losses import DiceLoss
        from segmentation_models.metrics import iou_score
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
        import cv2
        import numpy as np
        from paddleocr import PaddleOCR
        import time
        import re

        def get_bounding_boxes(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))

            return boxes

        input_shape = (256, 256, 3)

        model = Unet(
            backbone_name='resnet34',
            input_shape=input_shape,
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid'
        )

        model.load_weights(r'C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\Unet\unet_model.keras')

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=DiceLoss(),
            metrics=[iou_score]
        )

        ocr = PaddleOCR(use_angle_cls=True)
        input_size = (256, 256)

        total_time = 0
        total_images = len(image_files)  # Make sure image_files is defined in your script
        results = []

        start_time = time.time()

        for img_path in image_files:
            original_image = cv2.imread(str(img_path))
            image = original_image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, input_size)
            image_resized = image_resized.astype(np.float32) / 255.0
            input_img = np.expand_dims(image_resized, axis=0)

            predicted_mask = model.predict(input_img)
            predicted_mask = (predicted_mask > 0.4).astype(np.uint8)
            predicted_mask = predicted_mask[0, :, :, 0]
            output_size = (800, 600)
            predicted_mask_upscaled = cv2.resize(predicted_mask, output_size, interpolation=cv2.INTER_NEAREST)

            bboxes = get_bounding_boxes(predicted_mask_upscaled)

            original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            original_image_eq = cv2.equalizeHist(original_image_gray)
            original_image = cv2.cvtColor(original_image_eq, cv2.COLOR_GRAY2BGR)

            masked_image = np.ones_like(original_image) * 255

            for bbox in bboxes:
                x, y, w, h = bbox
                masked_image[y:y+h, x:x+w] = original_image[y:y+h, x:x+w]

            results_ocr = ocr.ocr(masked_image)

            img_code = ""

            if results:
                for result in results:

                    if result is not None:
                        for line in result:

                      
                            text = line[1][0]
                            

                            img_code += text + " "

                    

            real_code = annotations.get(img_path.name, {}).get("text", "").replace("\n", " ")

            real_code = real_code.replace(" ", "")
            img_code = img_code.replace(" ", "")

            lev_distance = levenshtein_distance(img_code.strip(), real_code)
            cer = lev_distance / len(real_code) if len(real_code) > 0 else 1.0
            exact_match = img_code.strip() == real_code

            results.append({
                "image_path": str(img_path),
                "ocr_text": img_code.strip(),
                "real_code": real_code,
                "levenshtein_distance": lev_distance,
                "cer": cer,
                "exact_match": exact_match
            })

        total_time += time.time() - start_time
        
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
        print(f"Average CER: {sum(r['cer'] for r in results) / total_images * 100:.2f}%")
        print(f"Exact match rate: {sum(r['exact_match'] for r in results) / total_images * 100:.2f}%")
        print(f"Total images processed: {total_images}")

        # Append total time and average CER to results
        results.append({
            "total_time": total_time,
            "average_time_per_image": total_time / total_images,
            "average_cer": sum(r['cer'] for r in results) / total_images * 100,
            "exact_match_rate": sum(r['exact_match'] for r in results) / total_images * 100,
            "total_images": total_images
        })


        # Save results to JSON
        output_file = os.path.join(output_path, f"{method_name}_evaluation_results_split_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)   

    elif method_name == "yolo":

        
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from ultralytics import YOLO
        import argparse
        import time

        def load_image(image_path):
            """Loads an image from the given path and converts it to RGB."""
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise FileNotFoundError(f"Error: Unable to load image at {image_path}")
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        def load_model(model_path):
            """Loads the YOLO model from the given path."""
            return YOLO(model_path)

        def predict(model, image, conf_threshold=0.4, img_size=640):
            """Runs YOLO model prediction on the given image and extracts bounding boxes."""

            results = model.predict(source=image, conf=conf_threshold, imgsz=img_size)
            class_names = model.names 

            bboxes = []

            if not results: 
                return bboxes
            

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    class_name = class_names[cls] if cls in class_names else f"Unknown_{cls}"
                    bboxes.append([class_name, (x1, y1), (x2, y2)])
                    
            return bboxes
        
        def ordenar_letras(bboxes, y_threshold=15):
            """
            Ordena las letras detectadas por líneas (eje Y) y luego por columna (eje X).
            :param bboxes: Lista con [class_name, (x1, y1), (x2, y2)]
            :param y_threshold: Tolerancia para agrupar letras en la misma línea
            :return: Lista de letras ordenadas como string
            """

            # Calcular centro de cada letra
            letras = []
            for class_name, (x1, y1), (x2, y2) in bboxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                letras.append((class_name, cx, cy))

            # Ordenar primero por Y (líneas de texto)
            letras.sort(key=lambda x: x[2])  # x[2] = centro Y

            # Agrupar por líneas usando un threshold
            lineas = []
            linea_actual = []
            for letra in letras:
                if not linea_actual:
                    linea_actual.append(letra)
                else:
                    if abs(letra[2] - linea_actual[-1][2]) < y_threshold:
                        linea_actual.append(letra)
                    else:
                        lineas.append(linea_actual)
                        linea_actual = [letra]
            if linea_actual:
                lineas.append(linea_actual)

            # Ordenar letras dentro de cada línea por X (horizontal)
            texto_final = ""
            for linea in lineas:
                linea_ordenada = sorted(linea, key=lambda x: x[1])  # x[1] = centro X
                texto_linea = "".join([letra[0] for letra in linea_ordenada])
                texto_final += texto_linea  # No se agregan espacios

            return texto_final


        model_path = r"C:\Users\User\OneDrive\Escritorio\TFG\Stage 2 - Text Detection\Code\Evaluation\YOLO\noSynthetic.pt"
        model = load_model(model_path)

        results = []
        total_time = 0
        total_images = len(image_files)
        start_time = time.time()

        for img_path in image_files:

            image_rgb = load_image(img_path)
            yolo_output = predict(model, image_rgb)

            ocr_text = ordenar_letras(yolo_output)
   
            real_code = annotations.get(img_path.name, {}).get("text", "").replace("\n", " ")

            real_code = real_code.replace(" ", "")
            ocr_text = ocr_text.replace(" ", "")

            lev_distance = levenshtein_distance(ocr_text, real_code)
            # print(f"Levenshtein distance: {lev_distance}")
        
            if len(real_code) > 0:
                cer = lev_distance / len(real_code)
            else:
                cer = 1.0 

            exact_match = ocr_text == real_code
            # print(f"Exact match: {exact_match}")

            result = {
                "image_path": str(img_path),
                "ocr_text": ocr_text,
                "real_code": real_code,
                "levenshtein_distance": lev_distance,
                "cer": cer,
                "exact_match": exact_match
            }
            results.append(result)
        
        total_time += time.time() - start_time
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
        print(f"Average CER: {sum(r['cer'] for r in results) / total_images * 100:.2f}%")
        print(f"Exact match rate: {sum(r['exact_match'] for r in results) / total_images * 100:.2f}%")
        print(f"Total images processed: {total_images}")

        # Append total time and average CER to results
        results.append({
            "total_time": total_time,
            "average_time_per_image": total_time / total_images,
            "average_cer": sum(r['cer'] for r in results) / total_images * 100,
            "exact_match_rate": sum(r['exact_match'] for r in results) / total_images * 100,
            "total_images": total_images
        })

        # Save results to JSON
        output_file = os.path.join(output_path, f"{method_name}_evaluation_results_split_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate OCR Detection Algorithm")

    parser.add_argument("method_name", type=str, help="Name of the detection method")
    parser.add_argument("model_path", type=str, nargs="?", default=None, help="Path to the model (if applicable)")
    parser.add_argument("output_path", type=str, nargs="?", default="./results", help="Path to save evaluation results")

    args = parser.parse_args()
    evaluate(args.method_name, split="real", model_path=args.model_path, output_path=args.output_path)



