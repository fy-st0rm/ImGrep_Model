import os
import torch
from torchvision import transforms
from PIL import Image
from model import VisionTransformer, TextEncoder, BPETokenizer
from dotenv import load_dotenv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the .env file from the current directory
load_dotenv()

# Parameters
IMG_DIR = os.getenv("TEST_IMG")
EPOCHS = int(os.getenv("EPOCHS"))
MODEL_PATH = f"clip_epoch_{EPOCHS}.pt"
MAX_LEN = int(os.getenv("MAX_LEN"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

# Load tokenizer
tokenizer = BPETokenizer("vocab.json", "merges.txt")
vocab_size = len(tokenizer.vocab)

# Load models
image_encoder = VisionTransformer()
text_encoder = TextEncoder(vocab_size)
# Assuming your full model combines both, or separate
# You may have a combined model class, if so, load that

image_encoder.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"], strict=False)
text_encoder.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"], strict=False)

image_encoder.to(device).eval()
text_encoder.to(device).eval()

# Image transform (same as training)
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load and encode all test images once
def load_images(img_dir):
	images = []
	image_names = []
	for fname in os.listdir(img_dir):
		if fname.lower().endswith((".jpg", ".jpeg", ".png")):
			path = os.path.join(img_dir, fname)
			try:
				img = Image.open(path).convert("RGB")
				img = transform(img)
				images.append(img)
				image_names.append(fname)
			except:
				print(f"Failed to load {fname}")
	return torch.stack(images), image_names

def encode_images(images):
	with torch.no_grad():
		images = images.to(device)
		embeddings = image_encoder(images)  # (N, embed_dim)
		embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
	return embeddings.cpu()

def encode_text(text):
	token_ids = tokenizer.encode(text)[:MAX_LEN]
	token_ids += [0] * (MAX_LEN - len(token_ids))
	token_ids = torch.tensor([token_ids]).to(device)
	with torch.no_grad():
		text_emb = text_encoder(token_ids)  # (1, embed_dim)
		text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
	return text_emb.cpu()

def search(text, image_embeddings, image_names, top_k=3):
	text_emb = encode_text(text)
	# Cosine similarity
	similarities = (image_embeddings @ text_emb.T).squeeze(1)  # (N,)

	top_scores, top_idxs = torch.topk(similarities, top_k)
	results = [(image_names[i.item()], top_scores[idx].item()) for idx, i in enumerate(top_idxs)]
	return results

def main():
	print("Loading images...")
	images, image_names = load_images(IMG_DIR)
	print(f"Loaded {len(images)} images")

	print("Encoding images...")
	image_embeddings = encode_images(images)

	while True:
		query = input("Enter search text (or 'quit' to exit): ").strip()
		if query.lower() == "quit":
			break
		results = search(query, image_embeddings, image_names)
		print("Top matches:")
		for fname, score in results:
			print(f"Image: {fname} | Similarity: {score:.4f}")

if __name__ == "__main__":
	main()
