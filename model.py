"""
model.py
MIL model implementations:
1. AttentionMIL - Embedding-level approach with gated attention
2. AdditiveMIL - Instance-level approach with attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================================
# ATTENTION MIL (Embedding-Level Approach)
# ============================================================================

class AttentionMIL_InstanceEmbedder(nn.Module):
    """
    Instance embedder with gated attention for Attention MIL.
    Transforms patches to weighted embeddings using ResNet18 + gated attention.
    """
    
    def __init__(self, embed_dim=512, attn_dim=128):
        super().__init__()
        # ResNet18 feature extractor
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1]  # Remove final FC layer
        self.feature_extractor = nn.Sequential(*modules)
        self.feature_dim = backbone.fc.in_features  # 512 for ResNet18
        
        # Gated attention parameters
        self.V = nn.Linear(self.feature_dim, attn_dim)
        self.U = nn.Linear(self.feature_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: [K, C, H, W] bag of K patches
            
        Returns:
            h: [K, feature_dim] weighted embeddings
        """
        K = x.size(0)
        
        # Extract features for each patch
        r = self.feature_extractor(x)  # [K, 512, 1, 1]
        r = r.view(K, -1)              # [K, 512]
        
        # Gated attention: a_i = softmax(w^T(tanh(Vr) ⊙ sigmoid(Ur)))
        a = self.w(torch.tanh(self.V(r)) * torch.sigmoid(self.U(r)))  # [K, 1]
        a = torch.softmax(a, dim=0)    # [K, 1]
        
        # Weighted embeddings: h_i = a_i * r_i
        h = a * r  # [K, 512]
        
        return h


class AttentionMIL_MILPool(nn.Module):
    """Sum pooling over weighted embeddings."""
    
    def forward(self, h):
        """
        Args:
            h: [K, feature_dim] weighted embeddings
            
        Returns:
            H: [feature_dim] bag-level representation
        """
        H = torch.sum(h, dim=0)  # [feature_dim]
        return H


class AttentionMIL_BagClassifier(nn.Module):
    """Bag-level classifier for Attention MIL."""
    
    def __init__(self, input_dim=512, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, H):
        """
        Args:
            H: [feature_dim] bag-level representation
            
        Returns:
            logits: [num_classes] bag-level prediction
        """
        return self.fc(H)


class AttentionMIL(nn.Module):
    """
    Full Attention MIL model: θ(X) = g ∘ σ ∘ f
    Embedding-level approach with gated attention pooling.
    """
    
    def __init__(self, embed_dim=512, attn_dim=128, num_classes=1):
        super().__init__()
        self.f = AttentionMIL_InstanceEmbedder(embed_dim=embed_dim, attn_dim=attn_dim)
        self.sigma = AttentionMIL_MILPool()
        self.g = AttentionMIL_BagClassifier(input_dim=embed_dim, num_classes=num_classes)
    
    def forward(self, x):
        """
        Args:
            x: [K, C, H, W] bag of K patches
            
        Returns:
            logits: [num_classes] bag-level prediction logits
        """
        h = self.f(x)       # [K, embed_dim] weighted embeddings
        H = self.sigma(h)   # [embed_dim] bag representation
        y = self.g(H)       # [num_classes] bag logits
        return y


# ============================================================================
# ADDITIVE MIL (Instance-Level Approach)
# ============================================================================

class AdditiveMIL_InstanceEmbedder(nn.Module):
    """
    Instance embedder for Additive MIL.
    Transforms patches to class-level logits with attention weighting.
    """
    
    def __init__(self, embed_dim=512, attn_dim=128, num_classes=1):
        super().__init__()
        # ResNet18 feature extractor
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.feature_extractor = nn.Sequential(*modules)
        self.feature_dim = resnet.fc.in_features  # 512 for ResNet18
        
        # Attention MLP ψ_m to compute α_i
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        
        # Final MLP ψ_p to map features to class logits
        self.final_mlp = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: [K, C, H, W] bag of K patches
            
        Returns:
            h_i: [K, num_classes] class logits per patch
        """
        K = x.shape[0]
        
        # Extract features
        feats = self.feature_extractor(x)  # [K, 512, 1, 1]
        feats = feats.view(K, -1)          # [K, 512]
        
        # Attention weights α_i = softmax(ψ_m(r_i))
        attn_logits = self.attn_mlp(feats)       # [K, 1]
        alpha = F.softmax(attn_logits, dim=0)    # [K, 1]
        
        # Weighted features
        weighted_feats = alpha * feats     # [K, 512]
        
        # Final per-patch class logits h_i = ψ_p(α_i * r_i)
        h_i = self.final_mlp(weighted_feats)  # [K, num_classes]
        
        return h_i


class AdditiveMIL_MILPool(nn.Module):
    """Sum pooling over instance logits."""
    
    def forward(self, h):
        """
        Args:
            h: [K, num_classes] instance logits
            
        Returns:
            H: [1, num_classes] bag-level logits
        """
        H = h.sum(dim=0, keepdim=True)  # [1, num_classes]
        return H


class AdditiveMIL_BagClassifier(nn.Module):
    """Identity function for Additive MIL (no additional transformation)."""
    
    def forward(self, H):
        return H


class AdditiveMIL(nn.Module):
    """
    Full Additive MIL model: θ(X) = g ∘ σ ∘ f
    Instance-level approach where bag logits = sum of instance logits.
    """
    
    def __init__(self, embed_dim=512, attn_dim=128, num_classes=1):
        super().__init__()
        self.f = AdditiveMIL_InstanceEmbedder(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            num_classes=num_classes
        )
        self.sigma = AdditiveMIL_MILPool()
        self.g = AdditiveMIL_BagClassifier()

    def forward(self, x):
        """
        Args:
            x: [K, C, H, W] bag of K patches
            
        Returns:
            logits: [1, num_classes] bag-level prediction logits
        """
        h_i = self.f(x)       # [K, num_classes] instance logits
        H = self.sigma(h_i)   # [1, num_classes] bag logits
        out = self.g(H)       # identity
        return out


# ============================================================================
# Model Factory
# ============================================================================

def create_mil_model(model_type='attention', embed_dim=512, attn_dim=128, num_classes=1):
    """
    Factory function to create MIL models.
    
    Args:
        model_type: 'attention' or 'additive'
        embed_dim: Embedding dimension
        attn_dim: Attention dimension
        num_classes: Number of output classes
        
    Returns:
        MIL model instance
    """
    if model_type.lower() == 'attention':
        return AttentionMIL(embed_dim=embed_dim, attn_dim=attn_dim, num_classes=num_classes)
    elif model_type.lower() == 'additive':
        return AdditiveMIL(embed_dim=embed_dim, attn_dim=attn_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'attention' or 'additive'.")


if __name__ == "__main__":
    print("Testing MIL Models...")
    
    # Create dummy input: 10 patches of size 128x128
    dummy_input = torch.randn(10, 3, 128, 128)
    
    print("\n1. Testing AttentionMIL:")
    model_attn = AttentionMIL(embed_dim=512, attn_dim=128, num_classes=1)
    output_attn = model_attn(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_attn.shape}")
    print(f"   Output value: {output_attn.item():.4f}")
    
    print("\n2. Testing AdditiveMIL:")
    model_add = AdditiveMIL(embed_dim=512, attn_dim=128, num_classes=1)
    output_add = model_add(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_add.shape}")
    print(f"   Output value: {output_add.item():.4f}")
    
    print("\n✓ Both models working correctly!")