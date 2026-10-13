"""Pipeline 'feature_extraction': obrazy -> wektory cech (cache w data/04_feature)."""

from kedro.pipeline import Node, Pipeline

from erli_machine_vision.model_zoo import BACKBONES, CLIP_BACKBONES

from .nodes import (
    encode_class_prompts,
    extract_embeddings,
    extract_handcrafted_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        Node(
            extract_handcrafted_features,
            inputs=["model_input_table", "params:feature_extraction.handcrafted"],
            outputs="handcrafted_features",
            name="extract_handcrafted_features",
        )
    ]
    nodes += [
        Node(
            extract_embeddings,
            inputs=[
                "model_input_table",
                f"params:feature_extraction.backbones.{name}",
                "params:feature_extraction.runtime",
            ],
            outputs=f"{name}_features",
            name=f"extract_{name}_features",
        )
        for name in BACKBONES
    ]
    nodes += [
        Node(
            encode_class_prompts,
            inputs=[
                f"params:feature_extraction.backbones.{name}",
                "params:feature_extraction.zero_shot_prompts",
            ],
            outputs=f"{name}_text_embeddings",
            name=f"encode_{name}_prompts",
        )
        for name in CLIP_BACKBONES
    ]
    return Pipeline(nodes)
