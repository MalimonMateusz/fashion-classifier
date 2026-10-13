"""Pipeline 'training': cechy -> klasyfikatory (data/06_models)."""

from kedro.pipeline import Node, Pipeline

from erli_machine_vision.model_zoo import BACKBONES, CLIP_BACKBONES

from .nodes import build_zero_shot, train_handcrafted_svm, train_linear_probe


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        Node(
            train_handcrafted_svm,
            inputs=[
                "handcrafted_features",
                "model_input_table",
                "params:training.handcrafted_svm",
            ],
            outputs="handcrafted_svm_model",
            name="train_handcrafted_svm",
        )
    ]
    nodes += [
        Node(
            train_linear_probe,
            inputs=[
                f"{bb}_features",
                "model_input_table",
                "params:training.linear_probe",
            ],
            outputs=f"{bb}_probe_model",
            name=f"train_{bb}_probe",
        )
        for bb in BACKBONES
    ]
    nodes += [
        Node(
            build_zero_shot,
            inputs=f"{bb}_text_embeddings",
            outputs=f"{bb}_zeroshot_model",
            name=f"build_{bb}_zeroshot",
        )
        for bb in CLIP_BACKBONES
    ]
    return Pipeline(nodes)
