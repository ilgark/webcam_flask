from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import mediapipe as mp
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/background'

# Initialize Mediapipe Selfie Segmentationsaaa
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if 'frame' not in data:
        return jsonify({"error": "No frame data"}), 400

    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    background_image_path = os.path.join('static', 'background')
    background_images = [f for f in os.listdir(background_image_path) if os.path.isfile(os.path.join(background_image_path, f))]

    if background_images:
        new_background = cv2.imread(os.path.join(background_image_path, background_images[0]))
        if new_background is None:
            return jsonify({"error": "Could not load background image"}), 500
        new_background_resized = cv2.resize(new_background, (frame.shape[1], frame.shape[0]))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)

        mask = results.segmentation_mask
        condition = mask > 0.5

        output_frame = np.where(condition[..., None], frame, new_background_resized)
    else:
        output_frame = frame

    _, buffer = cv2.imencode('.jpg', output_frame)
    output_frame_data = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"frame": output_frame_data})

if __name__ == '__main__':
    app.run(debug=True)
