import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Function to load your trained ResNet18 model (with 18 output classes)
def load_model(model_path):
    num_classes = 18  # Make sure this matches your training
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing function (resize, tensor conversion, and normalization)
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Dictionary containing the bio info for each animal
animal_info = {
    "Beaver": """**Habitat/Climate**: Freshwater rivers, streams, and lakes in temperate forests.  
**Location**: North America and parts of Europe.  

**Fun Facts**:  
- Beavers build dams that can be seen from space.  
- Their teeth never stop growing!
""",
    "Black Bear": """**Habitat/Climate**: Forests, swamps, and mountains in temperate climates.  
**Location**: North America.  

**Fun Facts**:  
- They are excellent tree climbers.  
- Black bears can be brown, cinnamon, or even white!
""",
    "Bobcat": """**Habitat/Climate**: Woodlands, deserts, and swamps in temperate to semi-arid regions.  
**Location**: North America.  

**Fun Facts**:  
- Their ear tufts help with hearing.  
- They’re mostly nocturnal and super sneaky hunters.
""",
    "Coyote": """**Habitat/Climate**: Plains, forests, mountains, and even cities; adapt to most climates.  
**Location**: North and Central America.  

**Fun Facts**:  
- Coyotes communicate with a variety of yips and howls.  
- They’re known to be clever and highly adaptable.
""",
    "Elephant": """**Habitat/Climate**: Grasslands, forests, and savannas in tropical and subtropical climates.  
**Location**: Africa and Asia.  

**Fun Facts**:  
- They recognize themselves in mirrors (self-aware!).  
- Their trunks have over 40,000 muscles!
""",
    "Goose": """**Habitat/Climate**: Wetlands, lakes, and rivers in temperate climates.  
**Location**: Found on every continent except Antarctica.  

**Fun Facts**:  
- They fly in a V formation to save energy.  
- Geese are super loyal to their mates.
""",
    "Gray Fox": """**Habitat/Climate**: Woodlands, brushy areas, and rocky terrain in warm and temperate zones.  
**Location**: North and Central America.  

**Fun Facts**:  
- They’re one of the few canines that can climb trees.  
- They’re mostly active at night.
""",
    "Horse": """**Habitat/Climate**: Grasslands and plains in temperate to dry climates.  
**Location**: Worldwide (domesticated), wild horses mostly in the Americas and Central Asia.  

**Fun Facts**:  
- Horses sleep both standing up and lying down.  
- They can’t vomit.
""",
    "Lion": """**Habitat/Climate**: Grasslands, savannas, and woodlands in warm climates.  
**Location**: Sub-Saharan Africa and a small population in India.  

**Fun Facts**:  
- A lion’s roar can be heard from 5 miles away.  
- They’re the only big cats that live in groups (prides).
""",
    "Mink": """**Habitat/Climate**: Wetlands and forests near rivers and streams in cool temperate zones.  
**Location**: North America and Europe.  

**Fun Facts**:  
- Minks are excellent swimmers.  
- They have super soft fur (which is why they’re often farmed, sadly).
""",
    "Mouse": """**Habitat/Climate**: Grasslands, forests, and urban areas in nearly all climates.  
**Location**: Worldwide.  

**Fun Facts**:  
- Mice can squeeze through holes as small as a dime.  
- They communicate with high-pitched sounds we can't hear.
""",
    "Mule Deer": """**Habitat/Climate**: Forests, deserts, and mountains in dry to temperate regions.  
**Location**: Western North America.  

**Fun Facts**:  
- Named for their large, mule-like ears.  
- Their antlers fork instead of branching from a single stem.
""",
    "Otter": """**Habitat/Climate**: Freshwater rivers and coastal marine environments in cool to temperate climates.  
**Location**: North America, Europe, Asia, and Africa.  

**Fun Facts**:  
- Otters hold hands when they sleep so they don’t drift apart.  
- They use rocks to crack open shellfish.
""",
    "Raccoon": """**Habitat/Climate**: Forests, marshes, and urban areas in temperate climates.  
**Location**: North and Central America.  

**Fun Facts**:  
- Raccoons are known for their “masked” faces.  
- They have super dexterous front paws.
""",
    "Rat": """**Habitat/Climate**: Found in almost all environments, from sewers to forests to homes.  
**Location**: Worldwide.  

**Fun Facts**:  
- Rats can laugh when tickled.  
- They’re insanely good swimmers and can tread water for days.
""",
    "Skunk": """**Habitat/Climate**: Woodlands, grasslands, and suburbs in temperate climates.  
**Location**: North and Central America.  

**Fun Facts**:  
- Their spray can be smelled from up to a mile away.  
- They usually give a “warning dance” before spraying.
""",
    "Turkey": """**Habitat/Climate**: Forests, grasslands, and fields in temperate zones.  
**Location**: North America.  

**Fun Facts**:  
- Wild turkeys can fly (short distances).  
- Their gobble can be heard a mile away.
""",
    "Western Gray Squirrel": """**Habitat/Climate**: Oak woodlands and conifer forests in mild to warm climates.  
**Location**: Western U.S. (especially California, Oregon).  

**Fun Facts**:  
- They use their tails like umbrellas.  
- They bury nuts to eat later but forget some—helping trees grow!
"""
}

# Define the classes list in the same order as used during training:
classes = [
    "Beaver", "Black Bear", "Bobcat", "Coyote", "Elephant", "Goose",
    "Gray Fox", "Horse", "Lion", "Mink", "Mouse", "Mule Deer",
    "Otter", "Raccoon", "Rat", "Skunk", "Turkey", "Western Gray Squirrel"
]

def main():
    st.title("Animal Print Classifier")
    st.write("Upload an image of an animal print to get its classification and info.")

    MODEL_PATH = "models/animal_tracks_model20.pth"
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}.")
        return

    model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        input_tensor = transform_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            pred_index = output.argmax(dim=1).item()
        
        predicted_animal = classes[pred_index]
        st.write("Predicted Animal:", predicted_animal)
        info = animal_info.get(predicted_animal, "No information available.")
        st.write("Info:", info)

if __name__ == "__main__":
    main()
