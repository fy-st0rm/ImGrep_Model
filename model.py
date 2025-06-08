import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# ---------------------------
# BPE Tokenizer
# ---------------------------
class BPETokenizer:
	def __init__(self, vocab_file, merges_file):
		with open(vocab_file, 'r') as f:
			self.vocab = json.load(f)
		with open(merges_file, 'r') as f:
			merges = f.read().splitlines()[1:]  # skip header
		self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(merges)}
		self.cache = {}

	def get_pairs(self, word):
		pairs = set()
		for i in range(len(word) - 1):
			pairs.add((word[i], word[i + 1]))
		return pairs

	def bpe(self, token):
		if token in self.cache:
			return self.cache[token]

		word = list(token)
		pairs = self.get_pairs(word)

		while True:
			if not pairs:
				break

			# Find the lowest rank pair
			bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
			if bigram not in self.bpe_ranks:
				break

			first, second = bigram
			new_word = []
			i = 0

			while i < len(word):
				# Find the bigram and merge
				if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
					new_word.append(first + second)
					i += 2
				else:
					new_word.append(word[i])
					i += 1

			word = new_word
			pairs = self.get_pairs(word)

		# Cache and return
		self.cache[token] = word
		return word

	def encode(self, text):
		tokens = text.lower().strip().split()
		bpe_tokens = []
		for token in tokens:
			word = self.bpe(token)
			bpe_tokens.extend(word)
		ids = [self.vocab.get(t, 0) for t in bpe_tokens]
		return ids

# ---------------------------
# Patch Embedding
# ---------------------------
class PatchEmbedding(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
		super().__init__()
		self.num_patches = (img_size // patch_size) ** 2
		self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		x = self.proj(x)
		x = x.flatten(2).transpose(1, 2)
		return x

# ---------------------------
# Transformer Encoder Block
# ---------------------------
class TransformerEncoderBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
		super().__init__()
		self.norm1 = nn.LayerNorm(embed_dim)
		self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
		self.norm2 = nn.LayerNorm(embed_dim)
		hidden_dim = int(embed_dim * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, embed_dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		norm_x = self.norm1(x)
		attn_output, _ = self.attn(norm_x, norm_x, norm_x)
		x = x + attn_output
		x = x + self.mlp(self.norm2(x))
		return x

# ---------------------------
# Vision Transformer
# ---------------------------
class VisionTransformer(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, depth=12):
		super().__init__()
		self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
		num_patches = self.patch_embed.num_patches
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
		self.pos_drop = nn.Dropout(0.1)
		self.transformer_blocks = nn.Sequential(
			*[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
		)
		self.head = nn.Identity()

	def forward(self, x):
		B = x.size(0)
		x = self.patch_embed(x)
		cls_tokens = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)
		x = x.transpose(0, 1)
		x = self.transformer_blocks(x)
		x = x.transpose(0, 1)
		return self.head(x[:, 0])

# ---------------------------
# Text Encoder
# ---------------------------
class TextEncoder(nn.Module):
	def __init__(self, vocab_size, embed_dim=768, max_len=77, num_heads=12, depth=12):
		super().__init__()
		self.token_embed = nn.Embedding(vocab_size, embed_dim)
		self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
		self.drop = nn.Dropout(0.1)
		self.encoder_blocks = nn.Sequential(
			*[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
		)
		self.head = nn.Identity()

	def forward(self, token_ids):
		x = self.token_embed(token_ids) + self.pos_embed[:, :token_ids.size(1), :]
		x = self.drop(x)
		x = x.transpose(0, 1)
		x = self.encoder_blocks(x)
		x = x.transpose(0, 1)
		return self.head(x[:, 0])

# ---------------------------
# CLIP Model
# ---------------------------
class CLIPModel(nn.Module):
	def __init__(self, image_encoder, text_encoder):
		super().__init__()
		self.image_encoder = image_encoder
		self.text_encoder = text_encoder
		self.image_proj = nn.Identity()
		self.text_proj = nn.Identity()

	def forward(self, images, token_ids):
		img_feats = self.image_encoder(images)
		txt_feats = self.text_encoder(token_ids)
		img_feats = F.normalize(self.image_proj(img_feats), dim=-1)
		txt_feats = F.normalize(self.text_proj(txt_feats), dim=-1)
		logits_per_image = img_feats @ txt_feats.T
		logits_per_text = txt_feats @ img_feats.T
		return logits_per_image, logits_per_text
