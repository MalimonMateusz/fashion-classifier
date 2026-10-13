"""Pipeline 'deployment': najlepszy model -> ONNX + model_card -> weryfikacja w cv2.dnn."""

from kedro.pipeline import Node, Pipeline

from erli_machine_vision.model_zoo import MODELS

from .nodes import export_onnx, verify_cv2_dnn


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                export_onnx,
                inputs=[
                    "best_model_info",
                    "params:feature_extraction.backbones",
                    "params:deployment",
                    *[f"{name}_model" for name in MODELS],
                ],
                outputs=["fashion_classifier_onnx", "model_card"],
                name="export_onnx",
            ),
            Node(
                verify_cv2_dnn,
                inputs=[
                    "fashion_classifier_onnx",
                    "model_card",
                    "model_input_table",
                    "params:deployment",
                    *[f"{name}_predictions" for name in MODELS],
                ],
                outputs="cv2_dnn_verification",
                name="verify_cv2_dnn",
            ),
        ]
    )
