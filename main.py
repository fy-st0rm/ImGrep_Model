import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from model import VisionTransformer, TextEncoder, CLIPModel, BPETokenizer
import torch.nn.functional as F
import random
import os
from dotenv import load_dotenv

# Load the .env file from the current directory
load_dotenv()

# Selecting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
MAX_LEN = int(os.getenv("MAX_LEN"))

# Dataset + Loader
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = CocoCaptions(
	root=os.getenv("TRAIN_IMG"),
	annFile=os.getenv("TRAIN_CAPTIONS"),
	transform=transform
)

def collate_fn(batch):
	images, captions = zip(*batch)
	texts = [cap[0] for cap in captions]
	return torch.stack(images), texts


# This reduces the dataset
# subset_size = len(dataset) // 5  # or pick a fixed number like 10000
# subset_indices = random.sample(range(len(dataset)), subset_size)
# subset_dataset = Subset(dataset, subset_indices)

loader = DataLoader(
	dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	collate_fn=collate_fn
)

# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Models
tokenizer = BPETokenizer("vocab.json", "merges.txt")
vocab_size = len(tokenizer.vocab)
image_encoder = VisionTransformer().to(device)
text_encoder = TextEncoder(vocab_size).to(device)
model = CLIPModel(image_encoder, text_encoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
def clip_loss(logits_per_image, logits_per_text, temperature=0.07):
	target = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
	loss_i2t = F.cross_entropy(logits_per_image / temperature, target)
	loss_t2i = F.cross_entropy(logits_per_text / temperature, target)
	return (loss_i2t + loss_t2i) / 2

for epoch in range(EPOCHS):
	print(f"[LOG]: Started Epoch {epoch+1}")
	total_loss = 0

	for i, (images, texts) in enumerate(loader):
		print(f"  -Batch {i}")
		images = images.to(device)
		token_ids = [tokenizer.encode(t)[:MAX_LEN] for t in texts]
		token_ids = [t + [0] * (MAX_LEN - len(t)) for t in token_ids]
		token_ids = torch.tensor(token_ids).to(device)

		logits_per_image, logits_per_text = model(images, token_ids)
		loss = clip_loss(logits_per_image, logits_per_text)

		total_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	torch.save({
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"loss": total_loss / (i + 1)
	}, f"clip_epoch_{epoch+1}.pt")

	print(f"[LOG]: Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}\n")
