import random
import string

import cv2

from face_mask_detector import load_face_mask_detector_model, get_face_mask_detections
from face_mask_detector.lib import _get_box_label_from_prediction, _add_labeled_box_to_image

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def _display_image_with_face_mask_detections(image_path: str, confidence_threshold: float) -> str:
    # load the input image, clone it, and grab the image spatial dimensions
    image = cv2.imread(image_path)

    face_mask_detector_model = load_face_mask_detector_model()

    (locations, predictions) = get_face_mask_detections(
        face_mask_detector_model, image, confidence_threshold
    )

    # loop over the detected face locations and their corresponding
    # prediction
    for (box, prediction) in zip(locations, predictions):
        (start_x, start_y, end_x, end_y) = box

        box_label = _get_box_label_from_prediction(prediction)

        _add_labeled_box_to_image(
            image, box_label, start_x, start_y, end_x, end_y,
        )

    # show the output image
    image_path: str = UPLOAD_FOLDER + ''.join(random.choice(string.ascii_lowercase) for _ in range(5)) + '.png'
    cv2.imwrite(image_path, image)
    return image_path


def test_single_image(image_path: str) -> str:
    return _display_image_with_face_mask_detections(image_path=image_path, confidence_threshold=0.5)
