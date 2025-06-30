# LIBRARIES
import random
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel,UniPCMultistepScheduler
import torch
import string
from tqdm import tqdm
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import NewPal as ft
import json
from diffusers.utils import load_image, make_image_grid

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import argparse

# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion ControlNet")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--num_total_images", type=int, default=100, help="Total number of images to generate")
    parser.add_argument("--num_genimages_per_image", type=int, default=10, help="Number of generated images per input image")
    parser.add_argument("--controlnet_model_path", type=str, default="/mnt/DADES/home/jgarcia/CODE/6) STYLE TRANSFER/ConrolNet Training/Resized Model Checkpoints", help="Path to the ControlNet model")
    return parser.parse_args()

# CONSTANTS

RealImagesDir = "/mnt/DADES2/STELA/data"
RealImagesJsonDir = "/mnt/DADES2/STELA/data/STELA_DATASET.json"

# READ DATA

try:
    with open(RealImagesJsonDir, "r") as f: 
        Annotations = json.load(f)
        i = len(Annotations)
        print("JSON file loaded successfully.")
        print(f"Number of entries: {i}")

except FileNotFoundError:
    print("JSON file not found.")

# DEVICE SETUP

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # You can change the index if you have more GPUs
    print("CUDA Device Name:", torch.cuda.get_device_name(device))
    print("Memoria Total (MB):", torch.cuda.get_device_properties(0).total_memory // (1024 ** 2))
else:
    print("No CUDA device available")

# FUNCTIONS
def add_spaces(text):
    return ' '.join(list(text))

def CannyCode(code, x, y, mask_size, scale=1.0, digit_spacing=5):
    """
    Draws text with soft blur and applies a Canny edge detector.
    
    Parameters:
    - code: str - The text code to render
    - x, y: int - Starting position for the text
    - mask_size: (width, height) - Size of the blank mask
    - scale: float - Text scale (default 1.0)
    - digit_spacing: int - Extra spacing between digits
    """
    def drawTextSoftBlur(string, x, y, ix, iy, scale, color, icolor, digit_spacing=5, canvas_shape=(512, 512)):
        """Draws bold text with soft blur, tracks character positions."""

        h, w = canvas_shape[:2]
        drawn_chars = []  # 🌸 We'll store [char, [x, y]] here

        for i, c in enumerate(string):
            code_width = ft.getCodeWidth(c, scale)
            code_height = ft.getCodeHeight(c, scale) if hasattr(ft, "getCodeHeight") else 50 * scale

            # Check bounds before drawing
            if 0 <= x < w and 0 <= y < h and x + code_width < w and y + code_height < h:
                ft.drawCode(c, x, y, scale, color, alpha=0.8)
                ft.drawCode(c, x + 1, y + 1, scale, color, alpha=0.8)

                # 💕 Save character and bottom-left corner
                if c != " ":
                    drawn_chars.append([c, int(x), int(y)])

            extra_spacing = digit_spacing if c != " " else 0
            x += ix + code_width + extra_spacing
            y += iy

            if icolor is not None:
                color = tuple(np.clip(np.array(color) + np.array(icolor), 0, 255))

        return drawn_chars


        # Prepare mask
    mask = np.zeros(mask_size, dtype=np.uint8)

    # Prepare text lines
    code_lines = [add_spaces(line) for line in code.split('\n')]

    # Drawing setup
    color = (1, 1, 1)       # white 💮
    icolor = (0, 0, 0)      # no inner color change
    space = scale * -5 / 0.67

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_position([0, 0, 1, 1])
    ax.imshow(mask, cmap='gray', aspect='auto')
    ax.axis('off')

    cur_y = y
    json_data = {"chars": [], "code": code, "scale": scale, "file_name": None, "reverse": []}

    for line in code_lines:
        chars = drawTextSoftBlur(line, x, cur_y, space, 0, scale, color, icolor, digit_spacing, canvas_shape=mask_size)
        cur_y += scale * 80 / 1.08
        json_data["chars"].append(chars)

    json_data["chars"] = [item for sublist in json_data["chars"] for item in sublist]

    # Render and convert
    fig.canvas.draw()
    fused_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    # Image processing
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGBA2BGR)
    dilated_mask = cv2.dilate(fused_img, np.ones((3, 3), np.uint8), iterations=1)
    canny_result = cv2.Canny(dilated_mask, threshold1=10, threshold2=80)

    return canny_result, json_data

def Image2Canny(image, drawing_region, code, font_scale, inside_polygon= True, digit_spacing=5):

    # Mask the current ID code
    polygon_points = [(x, y) for label, x, y in drawing_region]
    mask = Image.new('L', image.size, 0) 
    ImageDraw.Draw(mask).polygon(polygon_points, fill=255)

    image_rgb = image.convert("RGB")
    blurred_image = image_rgb.filter(ImageFilter.GaussianBlur(radius=15))  
    final_image = Image.composite(blurred_image, image_rgb, mask)
    image_np = np.array(final_image)

    # Background canny
    gray_with_text = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    canny_background = cv2.Canny(gray_with_text, threshold1=10, threshold2=65)

    # Code canny
    # compute x-initial and y-initial so it is centered in the polygon
    if inside_polygon:

        space = font_scale * - 5 / 0.67

        code_lines = [add_spaces(line) for line in code.split('\n')]
        widths = []
        for line in code_lines:
            widths.append(sum(ft.getCodeWidth(c, font_scale) for c in line) + space * (len(line)))
                
        max_width = max(widths)
    
        x_initial = int((drawing_region[0][1] + drawing_region[2][1]) / 2) - int(max_width/2)
        y_initial = int((drawing_region[0][2] + drawing_region[2][2]) / 2)
    
    else:
        # Move so one or more digits fall outside the polygon
        x_initial = 170
        y_initial = 370

    # Generate canny code

    size = canny_background.shape[:2]
    canny_code, json_data = CannyCode(code, x_initial, y_initial, size, scale=font_scale, digit_spacing=digit_spacing)

    # Control canny generation
    canny_img = cv2.bitwise_or(canny_background, canny_code)
    canny_img = Image.fromarray(canny_img).convert("RGB")

    return image, Image.fromarray(canny_background).convert("RGB"), Image.fromarray(canny_code).convert("RGB"), canny_img, json_data

def generate_code_lines(num_lines=3, length=10, min_alpha=120):
    # Get list of supported characters from ft.codes
    available_chars = [chr(c) for c in sorted(ft.codes.keys())]

    # Extract alphabets from available_chars (assuming you want A-Z, a-z) and not accents or special characters
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    numbers = [str(i) for i in range(10)]
    alphabets.extend(numbers)

    non_alphabets = [c for c in available_chars if c not in alphabets]

    lines = []
    for _ in range(num_lines):
 
        line_chars = [random.choice(alphabets) for _ in range(length - 1)]
        line_chars.append(random.choice(non_alphabets))

        # Shuffle so alphabets are not always at start
        random.shuffle(line_chars)
        lines.append(''.join(line_chars))

    return '\n'.join(lines)

def decode_chars(chars, scale):
    """
    Decodes the chars list into a string by sorting based on y-coordinates (lines) first
    and then x-coordinates within each line.

    Args:
        chars (list): A list of characters with their x and y positions.
                      Each character is represented as a tuple (char, x, y).
        scale (float): Scale factor to adjust spacing threshold.

    Returns:
        str: The decoded string with line breaks as per the image layout.
    """
    line_threshold = 5  
    spacing_threshold = 48 * float(scale) 
    
    chars = sorted(chars, key=lambda c: c[2]) 

    lines = []
    current_line = []
    for i, char in enumerate(chars):
        if i == 0 or abs(char[2] - chars[i - 1][2]) <= line_threshold:
            current_line.append(char)
        else:
            lines.append(current_line)
            current_line = [char]
    lines.append(current_line)

    decoded_lines = []
    for line in lines:
        sorted_line = sorted(line, key=lambda c: c[1])
        line_str = ""
        for j in range(len(sorted_line)):
            if j > 0 and (sorted_line[j][1] - sorted_line[j - 1][1]) > spacing_threshold:
                line_str += " "  # Add space for large gaps
            line_str += str(sorted_line[j][0])
        decoded_lines.append(line_str)

    return '\n'.join(decoded_lines)

def load_pipeline(controlnet_model_path):
    """
    Load the Stable Diffusion ControlNet pipeline.
    """
    class SD3CannyImageProcessor(VaeImageProcessor):
        def __init__(self):
            super().__init__(do_normalize=False)
        def preprocess(self, image, **kwargs):
            image = super().preprocess(image, **kwargs)
            image = image * 255 * 0.5 + 0.5
            return image
        def postprocess(self, image, do_denormalize=True, **kwargs):
            do_denormalize = [True] * image.shape[0]
            image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
            return image

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    )

    pipe.to(device)

    return pipe

# MAIN FUNCTION

def main():

    args = parse_args()

    output_path = args.output_path
    num_total_images = args.num_total_images
    num_genimages_per_image = args.num_genimages_per_image
    controlnet_model_path = args.controlnet_model_path

    os.makedirs(output_path, exist_ok=True)

    # Load the pipeline
    print("Loading ControlNet pipeline...")
    pipe = load_pipeline(controlnet_model_path)

    data ={}

    print("Generating images...")
    for i in tqdm(range(num_total_images), desc="Generating images"):
     
        entry = random.choice(list(Annotations.values()))
        file = entry["file"]
        drawing_region = entry['reserve']
        scale = entry['scale']
        chars = entry['chars']
    
        full_path = os.path.join(RealImagesDir, *file.split('\\'))
        original_img = Image.open(full_path)

        # Generate code

        lines = random.randint(1, 2)
        length = random.randint(2,4)
    
        code= generate_code_lines(num_lines=lines, length=length)
       
        # Font scale
        font_scale = random.uniform(0.8, 2.0)

        # Create input
        _, _, _, control_img, json_data = Image2Canny(original_img, drawing_region, code, font_scale, inside_polygon=False)

        prompt = (
            "Grayscale close-up of a laser-printed label on industrial packaging."
        )

        generated_imgs = pipe(
            prompt, 
            image=control_img.convert("RGB").resize((800, 800)), 
            num_inference_steps=20,
            num_images_per_prompt= num_genimages_per_image,
            controlnet_conditioning_scale=1.3,
        ).images

        if not generated_imgs:
            print(f"No images were generated for entry {i}. Skipping...")
            continue
        
        for idx in range(len(generated_imgs)):
           generated_imgs[idx] = generated_imgs[idx].resize((800, 600), Image.LANCZOS)

        for j, img in enumerate(generated_imgs):

            # Save the generated image
            output_name = f"Image_{i:04d}_gen_{j:02d}.png"
            output_path_full = os.path.join(output_path, output_name)
            img.save(output_path_full)
            json_data["file_name"] = output_name
            json_data["reverse"] = drawing_region
            data[len(data)] = json_data

    print("Image generation completed.")

    # Save the JSON data
    json_output_path = os.path.join(output_path, "generated_data.json")
    with open(json_output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
    print("Please run it directly to generate images.")
    print("Exiting...")
    exit(1)

# END OF SCRIPT
