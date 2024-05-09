import gradio as gr
import numpy as np
import cv2
import os
from PIL import Image

def colorize(file_path):
    # Create directories if they don't exist
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Check the file size
    if os.path.getsize(file_path) > 4 * 1024 * 1024:
        return None, "Error: File size exceeds 4 MB. Please upload a smaller file."
    
    # Save the input file in 'input' directory
    input_file_path = os.path.join('input', os.path.basename(file_path))
    os.rename(file_path, input_file_path)
    
    # Determine if it's an image or video
    if input_file_path.lower().endswith(('.mp4', '.avi')):
        return {
            "type": "video",
            "input": input_file_path,
            "output": process_video(input_file_path)[0]
        }
    else:
        return {
            "type": "image",
            "input": input_file_path,
            "output": process_image(input_file_path)[0]
        }

def process_video(input_file_path):
    # Video processing logic
    cap = cv2.VideoCapture(input_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_file_path = os.path.join('output', f"colorized_{os.path.basename(input_file_path)}")
    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        color_frame = colorize_frame(frame)
        out.write(color_frame)

    cap.release()
    out.release()
    return output_file_path, "Colorization successfull !"

def process_image(input_file_path):
    # Image processing logic
    img = Image.open(input_file_path)
    img_np = np.array(img)
    colorized_img_np = colorize_frame(img_np)
    colorized_img = Image.fromarray(colorized_img_np)
    
    output_file_path = os.path.join('output', f"colorized_{os.path.basename(input_file_path)}")
    colorized_img.save(output_file_path)
    return output_file_path, "Colorization successfull !"

def colorize_frame(np_image):
    img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # load our serialized black and white colorizer model and cluster
    # center points from disk
    #Note: Please take in account the directories of your local system.
    prototxt = "./models/models_colorization_deploy_v2.prototxt"
    model = "./models/colorization_release_v2.caffemodel"
    points = "./models/pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    # scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    # resize the Lab image to 224x224 (the dimensions the colorization
    #network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    # Return the colorized images
    return colorized


# Set up Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Colorize Black and White Images and Videos")
    gr.Markdown("Upload a black and white image or video. Ensure the file size is under 4MB.")
    with gr.Row():
        upload_button = gr.UploadButton(label="Upload Image or Video", file_count="single", type="filepath", file_types=["image", ".mp4", ".avi", ".mvi"])

    with gr.Row():
        input_image = gr.Image(label="Input Image", visible=True, type='filepath')
        output_image = gr.Image(label="Output Image", visible=True, type='filepath')

    with gr.Row():
        input_video = gr.Video(label="Input Video", visible=True)
        output_video = gr.Video(label="Output Video", visible=True)
    
    def handle_upload(file_path):
        result = colorize(file_path)
        print(result)
        if result["type"] == "video":
            return result["input"], result["output"], None, None
        return None, None, result["input"], result["output"]
        
    upload_button.upload(handle_upload, inputs=[upload_button], outputs=[input_video, output_video, input_image, output_image])

demo.launch()