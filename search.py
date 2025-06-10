import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import VisionTransformer, TextEncoder, CLIPModel, BPETokenizer
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()

# Parameters
IMG_DIR = os.getenv("TEST_IMG")
EPOCHS = int(os.getenv("EPOCHS"))
MODEL_PATH = f"clip_epoch_{EPOCHS}.pt"
MAX_LEN = int(os.getenv("MAX_LEN"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

class ImprovedCLIPSearch:
    def __init__(self, model_path, vocab_file="vocab.json", merges_file="merges.txt"):
        # Load tokenizer
        self.tokenizer = BPETokenizer(vocab_file, merges_file)
        vocab_size = len(self.tokenizer.vocab)

        # Initialize model components
        image_encoder = VisionTransformer()
        text_encoder = TextEncoder(vocab_size)
        self.model = CLIPModel(image_encoder, text_encoder)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(device).eval()

        # Image preprocessing (use ImageNet normalization for better results)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Cache for embeddings
        self.image_embeddings = None
        self.image_names = None

    def load_and_encode_images(self, img_dir, batch_size=32):
        print("Loading images...")
        images = []
        image_names = []

        # Load all images first
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                path = os.path.join(img_dir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    img = self.transform(img)
                    images.append(img)
                    image_names.append(fname)
                except Exception as e:
                    print(f"Failed to load {fname}: {e}")

        print(f"Loaded {len(images)} images")

        # Encode images in batches
        print("Encoding images...")
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = torch.stack(images[i:i+batch_size]).to(device)

                # Get image embeddings through the full model
                img_features = self.model.image_encoder(batch_images)
                img_features = self.model.image_proj(img_features)
                img_features = F.normalize(img_features, dim=-1)

                all_embeddings.append(img_features.cpu())

        self.image_embeddings = torch.cat(all_embeddings, dim=0)
        self.image_names = image_names
        print(f"Encoded {len(self.image_embeddings)} image embeddings")

    def encode_text(self, text):
        # Improved text preprocessing
        text = text.strip().lower()
        if not text:
            return None

        # Tokenize with proper padding/truncation
        token_ids = self.tokenizer.encode(text)[:MAX_LEN]
        if len(token_ids) == 0:
            token_ids = [0]  # fallback for empty tokenization

        # Pad to max length
        token_ids += [0] * (MAX_LEN - len(token_ids))
        token_ids = torch.tensor([token_ids]).to(device)

        with torch.no_grad():
            text_features = self.model.text_encoder(token_ids)
            text_features = self.model.text_proj(text_features)
            text_features = F.normalize(text_features, dim=-1)

        return text_features.cpu()

    def search(self, query, top_k=5, threshold=0.1):
        if self.image_embeddings is None:
            raise ValueError("Images not loaded. Call load_and_encode_images first.")

        text_embedding = self.encode_text(query)
        if text_embedding is None:
            return []

        # Compute similarities
        similarities = (self.image_embeddings @ text_embedding.T).squeeze(-1)

        # Apply threshold filter
        valid_indices = similarities > threshold
        if not valid_indices.any():
            print(f"No images found above similarity threshold {threshold}")
            # Return top results anyway but with warning
            valid_indices = torch.ones_like(similarities, dtype=torch.bool)

        filtered_similarities = similarities[valid_indices]
        filtered_names = [self.image_names[i] for i in range(len(self.image_names)) if valid_indices[i]]

        # Get top-k results
        if len(filtered_similarities) > 0:
            top_k = min(top_k, len(filtered_similarities))
            top_scores, top_indices = torch.topk(filtered_similarities, top_k)

            results = []
            for idx, score in zip(top_indices, top_scores):
                results.append({
                    'filename': filtered_names[idx],
                    'similarity': score.item(),
                    'confidence': 'high' if score.item() > 0.3 else 'medium' if score.item() > 0.15 else 'low'
                })
            return results

        return []

    def batch_search(self, queries, top_k=5):
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results

    def get_image_stats(self):
        if self.image_embeddings is None:
            return None

        return {
            'total_images': len(self.image_embeddings),
            'embedding_dim': self.image_embeddings.shape[1],
            'avg_norm': self.image_embeddings.norm(dim=-1).mean().item(),
            'embedding_std': self.image_embeddings.std().item()
        }


def main():
    # Initialize search system
    searcher = ImprovedCLIPSearch(MODEL_PATH)

    # Load and encode all images
    searcher.load_and_encode_images(IMG_DIR, batch_size=16)  # Adjust batch size based on GPU memory

    # Print stats
    stats = searcher.get_image_stats()
    if stats:
        print(f"\nDataset Stats:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Embedding dimension: {stats['embedding_dim']}")
        print(f"  Average embedding norm: {stats['avg_norm']:.4f}")
        print(f"  Embedding std: {stats['embedding_std']:.4f}")

    # Interactive search
    print("\n" + "="*50)
    print("CLIP Image Search Ready!")
    print("="*50)

    while True:
        query = input("\nEnter search query (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break

        if not query:
            continue

        print(f"\nSearching for: '{query}'")
        results = searcher.search(query, top_k=5, threshold=0.05)

        if results:
            print(f"\nTop {len(results)} matches:")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                confidence_emoji = "🟢" if result['confidence'] == 'high' else "🟡" if result['confidence'] == 'medium' else "🔴"
                print(f"{i}. {result['filename']}")
                print(f"   Similarity: {result['similarity']:.4f} {confidence_emoji} ({result['confidence']})")
        else:
            print("No matching images found. Try a different query.")

        # Option for batch search
        batch_input = input("\nEnter multiple queries separated by semicolons (or press Enter to continue): ").strip()
        if batch_input:
            queries = [q.strip() for q in batch_input.split(';') if q.strip()]
            if queries:
                print("\nBatch search results:")
                batch_results = searcher.batch_search(queries)
                for query, results in batch_results.items():
                    print(f"\n'{query}': {len(results)} matches")
                    for result in results[:3]:  # Show top 3 for each
                        print(f"  - {result['filename']} ({result['similarity']:.3f})")


if __name__ == "__main__":
    main()

