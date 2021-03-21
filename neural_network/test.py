import cv2

from face_mask_detector import load_face_mask_detector_model, get_face_mask_detections
from face_mask_detector.lib import _get_box_label_from_prediction, _add_labeled_box_to_image


def _display_image_with_face_mask_detections(image_path: str, confidence_threshold: float) -> None:
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
    cv2.imshow("Output", image)
    cv2.waitKey(0)


_display_image_with_face_mask_detections(image_path='test.jpg', confidence_threshold=0.5)
