# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def detect_and_label_shapes(image_path, output_path=None):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Compute the bounding box of the contour and use it to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Classify the shape
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif vertices > 4:
            # Check if the shape is a circle or an ellipse
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            if abs(1 - (area / circle_area)) < 0.2:
                shape_name = "Circle"
            else:
                shape_name = "Ellipse"
        else:
            shape_name = "Polyline"

        # Draw the contour and label the shape
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.putText(image, shape_name, (approx[0][0][0], approx[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Optionally save the output image
    if output_path:
        cv2.imwrite(output_path, image)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{filename}')
            file.save(file_path)

            # Process the image and save the result
            detect_and_label_shapes(file_path, output_path)
            return redirect(url_for('uploaded_file', filename=f'output_{filename}'))

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
