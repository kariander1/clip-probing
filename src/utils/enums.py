from enum import Enum
 
class TextModel(Enum):
    # CLIP Models
    CLIP_VIT_B_32 = 0  # openai/clip-vit-base-patch32
    CLIP_VIT_B_16 = 1  # SD1.5 (openai/clip-vit-base-patch16)
    CLIP_VIT_H_14 = 2  # SD2.1 / SD Turbo (laion/CLIP-ViT-H-14)
    CLIP_VIT_G_14 = 3  # SDXL / SDXL Turbo (laion/CLIP-ViT-G-14)
    CLIP_VIT_L_14 = 4  # SDXL Secondary Encoder (openai/clip-vit-large-patch14)
    
    # BERT Models
    BERT_BASE_UNCASED = 5  # bert-base-uncased
    BERT_LARGE_CASED = 6   # bert-large-cased
    
class Dataset_Type(Enum):
    RELATIONAL = 0
    RELATIONAL_POSITIONAL = 1
    RELATIONAL_POSITIONAL_LLM = 2
    COLORS_MULTILABEL = 3


class OptimizerName(Enum):
    ADAM = 0
    ADAMW = 1
    SGD = 2
    ADAMW_WITH_WEIGHT_DECAY_AND_MOMENTUM = 3
    SGD_WITH_WEIGHT_DECAY = 4
    SGD_WITH_MOMENTUM = 5
    SGD_WITH_WEIGHT_DECAY_AND_MOMENTUM = 6
