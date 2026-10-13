"""Rejestr porownywanych modeli: nazwa modelu -> nazwa zbioru cech, na ktorym dziala."""

BACKBONES = ["clip_vit_b32", "clip_vit_b16", "convnext_tiny", "efficientnet_b0"]
CLIP_BACKBONES = ["clip_vit_b32", "clip_vit_b16"]

MODELS: dict[str, str] = {
    "handcrafted_svm": "handcrafted",
    **{f"{bb}_zeroshot": bb for bb in CLIP_BACKBONES},
    **{f"{bb}_probe": bb for bb in BACKBONES},
}
