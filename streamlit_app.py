import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn as nn

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MMConvModel
class MMConvModel(nn.Module):
    def __init__(self, text_hidden_size=768, visual_hidden_size=256):
        super(MMConvModel, self).__init__()
        self.text_hidden_size = text_hidden_size
        self.visual_hidden_size = visual_hidden_size

        # Pre-trained BERT for text encoding
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")

        # Projection layer for visual features
        self.visual_proj = nn.Linear(visual_hidden_size, text_hidden_size)

        # Fusion layer for title and visual
        self.title_visual_fusion = nn.Linear(2 * text_hidden_size, text_hidden_size)

        # TransformerEncoder for fusion
        self.fusion = nn.TransformerEncoderLayer(d_model=text_hidden_size, nhead=8)

        # Final projection and classification layers
        self.proj = nn.Linear(2 * text_hidden_size, text_hidden_size)
        self.cls = nn.Linear(text_hidden_size, 1)

        # Activation and dropout
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, title, visual, conversation, text):
        # Encode title
        title_hidden = self.text_encoder(
            input_ids=title["input_ids"],
            attention_mask=title["attention_mask"]
        ).last_hidden_state[:, 0, :]  # [CLS]

        # Project visual features to text_hidden_size
        visual_hidden = self.visual_proj(visual)

        # Fuse title and visual
        fused_title_visual = torch.cat((title_hidden, visual_hidden), dim=1)
        fused_title_visual = self.title_visual_fusion(fused_title_visual)
        fused_title_visual = self.fusion(fused_title_visual.unsqueeze(0)).squeeze(0)

        # Encode target text
        text_hidden = self.text_encoder(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"]
        ).last_hidden_state[:, 0, :]

        # Final fusion: Concatenate fused title-visual and text hidden states
        final_hidden = torch.cat((fused_title_visual, text_hidden), dim=1)

        # Project back to text_hidden_size
        final_hidden = self.proj(final_hidden)

        # Apply TransformerEncoder for fusion
        fused_output = self.fusion(final_hidden.unsqueeze(0)).squeeze(0)

        # Classification
        logits = self.cls(self.dropout(self.gelu(fused_output)))
        return logits

# Load pre-trained MMConv model
model = MMConvModel().to(device)
checkpoint = torch.load("mmconv_model_train.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load Faster R-CNN for image feature extraction
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = faster_rcnn.backbone(image)["0"]
        features = features.mean(dim=(2, 3))
    return features

# Tokenize text
def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Streamlit UI
st.title("Dynamic Derailment Detection")
st.write("Upload an image, provide conversation details, and enter your comment to see if it derails the conversation in real-time.")

# Upload image
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    resized_image = image.resize((300, int(300 * image.height / image.width)))
    st.image(resized_image, caption="Uploaded Image (Resized)")
    image_features = preprocess_image(image)
else:
    image_features = None
    st.warning("Please upload an image to proceed.")

# Input fields
title = st.text_input("Enter the Conversation Title:", placeholder="Type your conversation title here...")
conversation_history = st.text_area("Enter Conversation History:", placeholder="Type each comment on a new line...")
typed_comment = st.text_input("Type your Target Comment:", placeholder="Type your comment to check if it derails...")

# Real-time prediction
if uploaded_image and title and conversation_history and typed_comment:
    conversation_list = conversation_history.split("\n")
    title_tokens = tokenize_texts([title]).to(device)
    conversation_tokens = tokenize_texts(conversation_list).to(device)
    target_tokens = tokenize_texts([typed_comment]).to(device)

    # Prepare inputs for the model
    title_input = {
        "input_ids": title_tokens["input_ids"],
        "attention_mask": title_tokens["attention_mask"]
    }
    target_input = {
        "input_ids": target_tokens["input_ids"],
        "attention_mask": target_tokens["attention_mask"]
    }

    with torch.no_grad():
        logits = model(
            title=title_input,
            visual=image_features,
            conversation=conversation_tokens["input_ids"],
            text=target_input
        )
        probabilities = torch.sigmoid(logits).item()
        prediction = "Derailment" if probabilities >= 0.5 else "Non-Derailment"

    # Display real-time prediction
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Derailment Probability:** {probabilities:.4f}")
else:
    st.info("Please fill out all fields and upload an image to see results.")