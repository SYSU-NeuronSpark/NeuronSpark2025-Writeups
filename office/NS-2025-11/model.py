import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        self.rgb_encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        self.pcd_encoder = models.resnet18(pretrained=True)
        self.pcd_encoder = nn.Sequential(*list(self.pcd_encoder.children())[:-1])        
        self.fusion = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        return

    def forward(self, x):
        batch_size, n_views, n_modality = x.shape[:3]
        features = []
        for v in range(n_views):
            rgb = x[:, v, 0]
            rgb_feat = self.rgb_encoder(rgb).flatten(1)
            
            pcd = x[:, v, 1]
            pcd_feat = self.pcd_encoder(pcd).flatten(1)
            
            fused = torch.cat([rgb_feat, pcd_feat], dim=1)
            fused = self.fusion(fused)
            features.append(fused)
        
        features = torch.stack(features, dim=1)
        return torch.mean(features, dim=1)

class LanguageEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8),
            num_layers=2
        )
        self.proj = nn.Linear(input_dim, hidden_dim)
        return
    
    def forward(self, x):
        x = self.transformer(x)
        return self.proj(x.mean(dim=1))

class MultiModalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_encoder = VisionEncoder()
        self.lang_encoder = LanguageEncoder()
        
        self.fusion = nn.Sequential(
            nn.Linear(512+256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.position_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 4)
        self.gripper_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return
    
    def forward(self, obs, instr):
        vis_feat = self.vis_encoder(obs)
        lang_feat = self.lang_encoder(instr)
        fused = torch.cat([vis_feat, lang_feat], dim=1)
        fused = self.fusion(fused)
        
        position = self.position_head(fused)
        rotation = self.rotation_head(fused)
        gripper = self.gripper_head(fused)
        return torch.cat([position, rotation, gripper], dim=1)

def hybrid_loss(pred, target):
        pos_loss = torch.nn.functional.mse_loss(pred[:, :3], target[:, :3])
        rot_loss = torch.nn.functional.mse_loss(pred[:, 3:7], target[:, 3:7])
        gripper_loss = torch.nn.functional.binary_cross_entropy(pred[:, 7], target[:, 7])
        return 0.7*pos_loss + 0.2*rot_loss + 0.1*gripper_loss