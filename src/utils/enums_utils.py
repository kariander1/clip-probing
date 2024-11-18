import torch.optim as optim
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from src.utils.enums import TextModel, Dataset_Type, OptimizerName
from src.data.dataset import MultiLabelRelationalDataset, SingleLabelRelationalDataset
    
def text_encoder_to_model_name(text_encoder: TextModel) -> str:
    paths = {
        TextModel.CLIP_VIT_B_32: "openai/clip-vit-base-patch32",
        TextModel.CLIP_VIT_B_16: "openai/clip-vit-base-patch16",  # SD1.5
        TextModel.CLIP_VIT_H_14: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",          # SD2.1 / SD Turbo
        TextModel.CLIP_VIT_G_14: "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",          # SDXL / SDXL Turbo
        TextModel.CLIP_VIT_L_14: "openai/clip-vit-large-patch14", # SDXL Secondary Encoder
        TextModel.BERT_BASE_UNCASED: "bert-base-uncased",
        TextModel.BERT_LARGE_CASED: "bert-large-cased",
    }

    if text_encoder not in paths:
        raise ValueError(f"Unknown text encoder: {text_encoder}")

    return paths[text_encoder]
    
def get_text_processor_and_model(text_encoder: TextModel, device="cuda"):
    model_name = text_encoder_to_model_name(text_encoder)
    model_class, tokenizer_class = text_encoder_classes(text_encoder)
    processor = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name).to(device)
    model.eval()
    
    if text_encoder == TextModel.CLIP_VIT_B_32:
        model = model.text_model
        tokenizer_max_length = processor.tokenizer.model_max_length
        hidden_size = model.config.hidden_size
    elif text_encoder == TextModel.CLIP_VIT_B_16:
        model = model.text_model
        tokenizer_max_length = processor.tokenizer.model_max_length
        hidden_size = model.config.hidden_size
    elif text_encoder == TextModel.CLIP_VIT_H_14:
        model = model.text_model
        tokenizer_max_length = processor.tokenizer.model_max_length
        hidden_size = model.config.hidden_size
    elif text_encoder == TextModel.CLIP_VIT_G_14:
        model = model.text_model
        tokenizer_max_length = processor.tokenizer.model_max_length
        hidden_size = model.config.hidden_size
    elif text_encoder == TextModel.CLIP_VIT_L_14:
        model = model.text_model
        tokenizer_max_length = processor.tokenizer.model_max_length
        hidden_size = model.config.hidden_size
    elif text_encoder == TextModel.BERT_BASE_UNCASED:
        tokenizer_max_length = processor.model_max_length
        hidden_size = model.config.hidden_size
    elif text_encoder == TextModel.BERT_LARGE_CASED:
        tokenizer_max_length = processor.model_max_length
        hidden_size = model.config.hidden_size
    else:
        raise ValueError(f"Unknown text model: {text_encoder}")

    return processor, model, tokenizer_max_length, hidden_size

def text_encoder_classes(text_encoder: TextModel):
    if text_encoder == TextModel.CLIP_VIT_B_32:
        return CLIPModel, CLIPProcessor
    if text_encoder == TextModel.CLIP_VIT_B_16:
        return CLIPModel, CLIPProcessor
    if text_encoder == TextModel.CLIP_VIT_H_14:
        return CLIPModel, CLIPProcessor
    if text_encoder == TextModel.CLIP_VIT_G_14:
        return CLIPModel, CLIPProcessor
    if text_encoder == TextModel.CLIP_VIT_L_14:
        return CLIPModel, CLIPProcessor
    if text_encoder == TextModel.BERT_BASE_UNCASED:
        return AutoModel, AutoTokenizer
    if text_encoder == TextModel.BERT_LARGE_CASED:
        return AutoModel, AutoTokenizer
    raise ValueError(f"Unknown text model: {text_encoder}")

def dataset_type_to_class(dataset_type: Dataset_Type):
    if dataset_type == Dataset_Type.COLORS_MULTILABEL:
        return MultiLabelRelationalDataset
    elif dataset_type == Dataset_Type.RELATIONAL:
        return SingleLabelRelationalDataset
    elif dataset_type == Dataset_Type.RELATIONAL_POSITIONAL:
        return SingleLabelRelationalDataset
    elif dataset_type == Dataset_Type.RELATIONAL_POSITIONAL_LLM:
        return SingleLabelRelationalDataset
    elif dataset_type == Dataset_Type.COUNT_AIRPLANES:
        return SingleLabelRelationalDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
def dataset_type_to_path(dataset_type: Dataset_Type) -> str:
    paths = {
        Dataset_Type.COLORS_MULTILABEL: "data/labeled_captions_by_color.json",
        Dataset_Type.RELATIONAL: "data/relation_prediction.json",
        Dataset_Type.RELATIONAL_POSITIONAL: "data/relational_dataset.json",
        Dataset_Type.RELATIONAL_POSITIONAL_LLM: "data/relational_dataset_llm.json",
        Dataset_Type.COUNT_AIRPLANES: "data/airplane_captions_training.json",
    }

    if dataset_type not in paths:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return paths[dataset_type]

def dataset_labels_are_super_labels(dataset_type: Dataset_Type) -> bool:
    if dataset_type == Dataset_Type.COLORS_MULTILABEL:
        return False
    elif dataset_type == Dataset_Type.RELATIONAL:
        return False
    elif dataset_type == Dataset_Type.RELATIONAL_POSITIONAL:
        return True
    elif dataset_type == Dataset_Type.RELATIONAL_POSITIONAL_LLM:
        return True
    elif dataset_type == Dataset_Type.COUNT_AIRPLANES:
        return False
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
def optimizer_name_to_class(optimizer_name: OptimizerName):
    if optimizer_name == OptimizerName.ADAM:
        return optim.Adam
    elif optimizer_name == OptimizerName.ADAMW:
        return optim.AdamW
    elif optimizer_name == OptimizerName.SGD:
        return optim.SGD
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
    
def dataset_type_to_criterion(dataset_type: Dataset_Type):
    if dataset_type == Dataset_Type.COLORS_MULTILABEL:
        return torch.nn.BCEWithLogitsLoss()
    elif dataset_type == Dataset_Type.RELATIONAL:
        return torch.nn.CrossEntropyLoss()
    elif dataset_type == Dataset_Type.RELATIONAL_POSITIONAL:
        return torch.nn.CrossEntropyLoss()
    elif dataset_type == Dataset_Type.RELATIONAL_POSITIONAL_LLM:
        return torch.nn.CrossEntropyLoss()
    elif dataset_type == Dataset_Type.COUNT_AIRPLANES:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")